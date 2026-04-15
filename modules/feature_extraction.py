import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)

def load_clean_data(csvPath):
    logger.info("Loading processed emails from: %s", csvPath)
    df = pd.read_csv(csvPath)

    start = len(df)
    df = df.dropna(subset=['sentence_text', 'lemmas'])
    df = df[df['lemmas'].str.strip().astype(bool)]

    dropped = start - len(df)
    logger.info("Loaded %d rows (%d empty)", len(df), dropped)
    return df

def get_tfidf(vocabSize=5000):
    return TfidfVectorizer(max_features=vocabSize, sublinear_tf=True, min_df=2, max_df=0.95, ngram_range=(1, 2))

def get_lda(numTopics=8):
    return LatentDirichletAllocation(n_components=numTopics, learning_method='online', random_state=42, max_iter=20)

def build_features(df, tfidf, lda):
    emailList = df['lemmas'].tolist()

    logger.info("Fitting TF-IDF on %d documents", len(emailList))
    tfidfMat = tfidf.fit_transform(emailList)

    logger.info("Fitting LDA topic model (%d topics)", lda.n_components)
    lda.fit(tfidfMat)

    tfidfArr = tfidfMat.toarray()
    ldaArr   = lda.transform(tfidfMat)

    logger.info("TF-IDF matrix shape: %s", tfidfArr.shape)
    logger.info("LDA topics shape: %s", ldaArr.shape)

    featureMat = np.hstack([tfidfArr, ldaArr])
    logger.info("Final matrix: %s", featureMat.shape)

    scaler = MinMaxScaler()
    featureMat = scaler.fit_transform(featureMat)

    labelEnc = LabelEncoder()
    labels = labelEnc.fit_transform(df['trueDomain'].values)

    return featureMat, labels, labelEnc, scaler

def transform_new_emails(emailList, tfidf, lda):
    if not emailList:
        raise ValueError("emailList is empty")

    tfidfVecs = tfidf.transform(emailList)
    ldaVecs   = lda.transform(tfidfVecs)
    return tfidfVecs.toarray(), ldaVecs