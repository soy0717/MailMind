import os
import sys
import logging 
import joblib
import pandas as pd
from modules.feature_extraction import load_clean_data, build_features, get_tfidf, get_lda
from modules.classifier import train_domain_classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

root_directory   = os.path.dirname(os.path.abspath(__file__))
csvp             = os.path.join(root_directory, 'data', 'processed_emails.csv')
models_directory = os.path.join(root_directory, 'models')
correctionsp = os.path.join(root_directory, 'ui', 'agent_state', 'corrections_training.csv')


def load_with_corrections(csvPath, correctionsPath):
    df = load_clean_data(csvPath)

    if os.path.exists(correctionsPath):
        corrections = pd.read_csv(correctionsPath)
        corrections['sentence_text'] = corrections['lemmas']
        numNew = len(corrections)
        df = pd.concat([df, corrections], ignore_index=True)
        logger.info("Mixed in %d user corrections for retraining", numNew)
    else:
        logger.info("No corrections file found, training on original data only")

    return df


def main():
    logger.info("Training starting")
    df = load_with_corrections(csvp, correctionsp)

    tfidf = get_tfidf()
    lda   = get_lda()
    X, y, label_enc, scaler = build_features(df, tfidf, lda)

    logger.info("Feature matrix: %s, %d classes", X.shape, len(label_enc.classes_))

    os.makedirs(models_directory, exist_ok=True)
    joblib.dump({'tfidf': tfidf, 'lda': lda}, os.path.join(models_directory, 'tfidf_lda.joblib'))
    joblib.dump(label_enc, os.path.join(models_directory, 'label_encoder.joblib'))
    joblib.dump(scaler,    os.path.join(models_directory, 'scaler.joblib'))
    logger.info("Saved to %s", models_directory)

    results = train_domain_classifier(X, y, labelList=list(label_enc.classes_), outDir=models_directory, confThresh=0.50)

    logger.info("Done. %d emails, %d features", len(y), X.shape[1])
    logger.info("accuracy=%.4f, f1=%.4f", results['accuracy'], results['macro_f1'])

if __name__ == '__main__':
    main()