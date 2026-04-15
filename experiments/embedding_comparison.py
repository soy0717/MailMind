import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_csv     = os.path.join(root_directory, 'data', 'processed_emails.csv')
output_directory   = os.path.join(root_directory, 'experiments', 'results')


def load_ablation_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['sentence_text', 'lemmas'])
    df = df[df['lemmas'].str.strip().astype(bool)]
    return df

def tfidf(texts):
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True, min_df=2, max_df=0.95, ngram_range=(1, 2))
    return vec.fit_transform(texts).toarray()


def word2vec(texts, vector_size=300):
    from gensim.models import Word2Vec
    tokenized = [t.split() for t in texts]
    model = Word2Vec(sentences=tokenized, vector_size=vector_size, window=5, min_count=2, sg=0, workers=4, seed=42)
    embeddings = []
    for words in tokenized:
        vecs = [model.wv[w] for w in words if w in model.wv]
        embeddings.append(np.mean(vecs, axis=0) if vecs else np.zeros(vector_size))
    return np.array(embeddings)


def build_glove(texts):
    import gensim.downloader as api
    model = api.load('glove-wiki-gigaword-300')
    embeddings = []
    for text in texts:
        words = text.split()
        vecs = [model[w] for w in words if w in model]
        embeddings.append(np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size))
    return np.array(embeddings)


def build_fasttext(texts):
    import gensim.downloader as api
    model = api.load('fasttext-wiki-news-subwords-300')
    embeddings = []
    for text in texts:
        words = text.split()
        vecs = [model[w] for w in words if w in model]
        embeddings.append(np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size))
    return np.array(embeddings)


def build_bert_cls(texts, batch_size=32):
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**encoded)
        embeddings.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)


def build_sbert(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def evaluate_embedding(name, X, y, label_names, n_splits=5):
    from imblearn.over_sampling import SMOTE

    logger.info("── Evaluating: %s (%d samples, %d features) ──", name, X.shape[0], X.shape[1])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1s, fold_accs = [], []

    all_preds = np.full(len(y), -1, dtype=int)
    all_confs = np.zeros(len(y))

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_train)) - 1))
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except ValueError:
            logger.warning("  Fold %d: SMOTE failed, using raw data", fold_idx + 1)
            X_train_res, y_train_res = X_train, y_train

        svm = LinearSVC(C=1.0, max_iter=10000, class_weight='balanced', random_state=42)
        calibrated = CalibratedClassifierCV(svm, cv=3)
        calibrated.fit(X_train_res, y_train_res)

        preds = calibrated.predict(X_test)
        probas = calibrated.predict_proba(X_test)
        confs = probas.max(axis=1)

        fold_f1s.append(f1_score(y_test, preds, average='macro'))
        fold_accs.append(accuracy_score(y_test, preds))

        all_preds[test_idx] = preds
        all_confs[test_idx] = confs

        logger.info("  Fold %d: Macro-F1=%.4f  Acc=%.4f", fold_idx + 1, fold_f1s[-1], fold_accs[-1])

    mean_f1  = np.mean(fold_f1s)
    mean_acc = np.mean(fold_accs)
    std_f1   = np.std(fold_f1s)

    low_conf_count = int((all_confs < 0.50).sum())
    low_conf_pct   = 100.0 * low_conf_count / len(y)

    logger.info("  RESULT: Macro-F1=%.4f (±%.4f)  Acc=%.4f  Low-conf=(%d, %.1f%%)",
                mean_f1, std_f1, mean_acc, low_conf_count, low_conf_pct)

    report = classification_report(y, all_preds, target_names=label_names, digits=4, zero_division=0)
    print(f"\n{'='*60}")
    print(f"  Classification Report: {name}")
    print(f"{'='*60}")
    print(report)

    return {
        'embedding': name,
        'macro_f1': round(mean_f1, 4),
        'f1_std': round(std_f1, 4),
        'accuracy': round(mean_acc, 4),
        'low_confidence_count': low_conf_count,
        'low_confidence_pct': round(low_conf_pct, 1),
        'dimensions': X.shape[1],
    }, all_preds, all_confs


def extract_gold_examples(df, y, label_enc, predictions_map, output_path):
    logger.info("Extracting Gold Examples...")

    if 'SBERT' not in predictions_map:
        logger.warning("SBERT predictions not available, skipping gold examples")
        return

    sbert_preds = predictions_map['SBERT']
    gold_rows = []

    for word_emb_name in ['TF-IDF', 'GloVe']:
        if word_emb_name not in predictions_map:
            continue

        word_preds = predictions_map[word_emb_name]

        for i in range(len(y)):
            true_label = label_enc.inverse_transform([y[i]])[0]
            word_pred  = label_enc.inverse_transform([word_preds[i]])[0] if word_preds[i] >= 0 else 'N/A'
            sbert_pred = label_enc.inverse_transform([sbert_preds[i]])[0] if sbert_preds[i] >= 0 else 'N/A'

            if word_preds[i] != y[i] and sbert_preds[i] == y[i]:
                gold_rows.append({
                    'email_id':       df.iloc[i]['email_id'],
                    'true_label':     true_label,
                    f'{word_emb_name}_pred': word_pred,
                    'sbert_pred':     sbert_pred,
                    'sentence_text':  str(df.iloc[i]['sentence_text'])[:300],
                    'comparison':     f'{word_emb_name} vs SBERT',
                })

    if gold_rows:
        gold_df = pd.DataFrame(gold_rows)
        gold_df.to_csv(output_path, index=False)
        logger.info("Saved %d gold examples to %s", len(gold_df), output_path)

        print(f"\n{'='*60}")
        print(f"  Gold Examples (Word Embedding fails, SBERT succeeds)")
        print(f"{'='*60}")
        for _, row in gold_df.head(5).iterrows():
            print(f"\n  [{row['comparison']}] True: {row['true_label']}")
            print(f"  Text: {row['sentence_text'][:120]}...")
    else:
        logger.info("No gold examples found")


def plot_ablation_chart(results_df, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
    names  = results_df['embedding'].tolist()
    scores = results_df['macro_f1'].tolist()
    stds   = results_df['f1_std'].tolist()

    bars = ax.bar(names, scores, color=colors[:len(names)], edgecolor='white',
                  linewidth=1.5, yerr=stds, capsize=5, alpha=0.9)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Macro-F1 Score', fontsize=13)
    ax.set_title('CIA2 Ablation Study — Embedding Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved ablation chart to %s", output_path)


def main():
    os.makedirs(output_directory, exist_ok=True)

    df = load_ablation_data(input_csv)

    label_enc  = LabelEncoder()
    y          = label_enc.fit_transform(df['trueDomain'].values)
    label_names = list(label_enc.classes_)

    lemmas_texts   = df['lemmas'].fillna('').tolist()
    sentence_texts = df['sentence_text'].fillna('').tolist()

    embedding_configs = [
        ('TF-IDF',   'word',     lambda: tfidf(lemmas_texts)),
        ('Word2Vec', 'word',     lambda: word2vec(lemmas_texts)),
        ('GloVe',    'word',     lambda: build_glove(lemmas_texts)),
        ('FastText', 'word',     lambda: build_fasttext(lemmas_texts)),
        ('BERT-CLS', 'sentence', lambda: build_bert_cls(sentence_texts)),
        ('SBERT',    'sentence', lambda: build_sbert(sentence_texts)),
    ]

    all_results    = []
    predictions_map = {}

    for name, emb_type, extractor_fn in embedding_configs:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  STARTING: %s (%s-level embedding)", name, emb_type)
        logger.info("=" * 60)

        try:
            X = extractor_fn()
            metrics, preds, confs = evaluate_embedding(name, X, y, label_names)
            all_results.append(metrics)
            predictions_map[name] = preds
        except Exception as e:
            logger.error("FAILED: %s — %s", name, str(e))
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        results_df   = pd.DataFrame(all_results)
        results_path = os.path.join(output_directory, 'ablation_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info("Saved ablation results to %s", results_path)

        print(f"\n{'='*60}")
        print(f"  ABLATION STUDY SUMMARY")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

        plot_ablation_chart(results_df, os.path.join(output_directory, 'ablation_chart.png'))

    extract_gold_examples(
        df, y, label_enc, predictions_map,
        os.path.join(output_directory, 'gold_examples.csv'),
    )

    logger.info("Ablation study complete!")


if __name__ == '__main__':
    main()
