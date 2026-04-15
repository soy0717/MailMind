import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

logger = logging.getLogger(__name__)

confidence_threshold = 0.50

def save_confusion_matrix(yTrue, yPred, labelNames, savePath):
    cm = confusion_matrix(yTrue, yPred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labelNames, yticklabels=labelNames, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Domain Classification — Confusion Matrix')
    plt.tight_layout()
    fig.savefig(savePath, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", savePath)

def train_domain_classifier(X, y, labelList=None, outDir='models', confThresh=confidence_threshold):
    os.makedirs(outDir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    logger.info("Train: %d  Test: %d", len(y_train), len(y_test))

    paramGrid = {'C': [0.01, 0.1, 1.0, 5.0, 10.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    baseSvm = LinearSVC(max_iter=5000, random_state=42)
    grid = GridSearchCV(baseSvm, paramGrid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    bestC = grid.best_params_['C']
    logger.info("Best C: %s, CV Macro-F1: %.4f", bestC, grid.best_score_)

    tunedSvm = LinearSVC(C=bestC, max_iter=5000, random_state=42)
    model = CalibratedClassifierCV(tunedSvm, cv=3)
    model.fit(X_train, y_train)

    yPred  = model.predict(X_test)
    yProba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, yPred)
    f1  = f1_score(y_test, yPred, average='macro')
    logger.info("Test Accuracy: %.4f", acc)
    logger.info("Test Macro-F1: %.4f", f1)

    report = classification_report(y_test, yPred, target_names=labelList, digits=4)
    print('\n' + report)

    reportPath = os.path.join('experiments', 'results', 'classification_report_tfidf_lda.txt')
    with open(reportPath, 'w') as f:
        f.write(report)

    save_confusion_matrix(y_test, yPred, labelList, os.path.join('experiments', 'results', 'confusion_matrix_tfidf_lda.png'))

    maxProbs = yProba.max(axis=1)
    lowConf  = (maxProbs < confThresh).sum()
    logger.info("Low-confidence predictions (< %.0f%%): %d / %d (%.1f%%)", 
                confThresh * 100, lowConf, len(y_test),
                100.0 * lowConf / len(y_test))

    modelPath = os.path.join(outDir, 'svm_model.joblib')
    joblib.dump(model, modelPath)
    logger.info("Model saved to %s", modelPath)

    return {
        'accuracy':             acc,
        'macro_f1':             f1,
        'best_C':               bestC,
        'low_confidence_count': int(lowConf),
        'model':                model,
        'classes':              model.classes_,
    }


def predict_domains(model, X, confThresh=confidence_threshold):
    preds  = model.predict(X)
    probas = model.predict_proba(X)
    confs  = probas.max(axis=1)
    needsReview = confs < confThresh
    return preds, confs, needsReview


def load_classifier(modelPath):
    model = joblib.load(modelPath)
    logger.info("Loaded model from %s", modelPath)
    return model