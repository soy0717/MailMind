import os
import json
import csv
import logging
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

RETRAIN_THRESHOLD = 10


def make_agent(model, label_encoder, state_dir='agent_state'):
    os.makedirs(state_dir, exist_ok=True)
    return {
        'model':            model,
        'label_encoder':    label_encoder,
        'state_dir':        state_dir,
        'keyword_rules':    defaultdict(Counter),
        'feedback_log':     [],
        'confirmation_log': [],
    }


def classify_email(agent, text, features):
    model    = agent['model']
    labelEnc = agent['label_encoder']
    kwRules  = agent['keyword_rules']

    if features.ndim == 1:
        features = features.reshape(1, -1)

    proba     = model.predict_proba(features)[0]
    svmIdx    = np.argmax(proba)
    svmConf   = proba[svmIdx]
    svmDomain = labelEnc.inverse_transform([svmIdx])[0]

    print(f"SVM Prediction: {svmDomain}  ({svmConf:.1%})")

    svmScores = {labelEnc.inverse_transform([i])[0]: p for i, p in enumerate(proba)}

    print('\nBase SVM Probabilities:')
    for idx in np.argsort(proba)[-3:][::-1]:
        d = labelEnc.inverse_transform([idx])[0]
        print(f"    {d}: {proba[idx]:.1%}")

    logOdds = {d: 0.0 for d in svmScores}
    if kwRules:
        logOdds = compute_log_odds(agent, text)

    print('\nKeyword weight addition:')
    finalLogProbs = {}
    for domain in svmScores:
        baseProb    = svmScores[domain]
        baseLogProb = np.log(max(1e-9, baseProb))
        kwEvidence  = logOdds.get(domain, 0.0)
        finalLogProbs[domain] = baseLogProb + kwEvidence

        if kwEvidence != 0:
            print(f"    {domain}: ln({baseProb:.3f}) + {kwEvidence:.2f} = {finalLogProbs[domain]:.3f}")

    # softmax
    maxLog     = max(finalLogProbs.values())
    expScores  = {d: np.exp(v - maxLog) for d, v in finalLogProbs.items()}
    totalExp   = sum(expScores.values())
    finalProbs = {d: s / totalExp for d, s in expScores.items()}

    bestDomain = max(finalProbs, key=finalProbs.get)
    finalConf  = finalProbs[bestDomain]

    print('\nFinal Probabilities:')
    for d, p in sorted(finalProbs.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"    {d}: {p:.1%}")

    needsReview = finalConf < 0.50
    source = 'ensemble' if any(v != 0 for v in logOdds.values()) else 'svm_model'

    print(f"\nFinal Decision: {bestDomain}  (source: {source})")
    print(f"Needs Review: {'Yes' if needsReview else 'No'}")

    return {
        'predicted_domain': bestDomain,
        'confidence':       float(finalConf),
        'needs_review':     needsReview,
        'source':           source,
        'svm_prediction':   svmDomain,
    }


def confirm_prediction(agent, emailId, text, domain, confidence):
    update_keyword_rules(agent, text, domain)

    record = {
        'timestamp':           datetime.now().isoformat(),
        'email_id':            emailId,
        'predicted':           domain,
        'correct':             domain,
        'text_snippet':        text[:200],
        'type':                'confirmation',
        'original_confidence': confidence,
    }
    agent['confirmation_log'].append(record)
    save_agent_state(agent)

    logger.info("Confirmation logged for %s domain=%s (conf=%.2f). Keywords reinforced.",
                emailId, domain, confidence)


def log_feedback(agent, emailId, text, predicted, correct):
    if predicted == correct:
        print(f"Prediction correct: {predicted}")
        logger.info("Feedback: prediction correct, no update needed.")
        return False

    print(f"Correction: {predicted} → {correct}")

    record = {
        'timestamp':    datetime.now().isoformat(),
        'email_id':     emailId,
        'predicted':    predicted,
        'correct':      correct,
        'text_snippet': text[:200],
    }
    agent['feedback_log'].append(record)
    logger.info("Feedback logged: %s → %s (was %s)", emailId, correct, predicted)

    append_correction_for_training(agent['state_dir'], emailId, text, correct)

    print(f"\nExtracting keywords...")
    update_keyword_rules(agent, text, correct)
    save_agent_state(agent)

    numCorrections = len(agent['feedback_log'])
    print(f"\nTotal corrections logged: {numCorrections}")

    if numCorrections >= RETRAIN_THRESHOLD:
        print(f"Reached {RETRAIN_THRESHOLD} corrections — run python main.py to retrain!")
        logger.info("Reached %d corrections — retrain recommended!", numCorrections)
        return True

    return False

def get_feedback_summary(agent):
    summary = {
        'total_corrections':   len(agent['feedback_log']),
        'total_confirmations': len(agent['confirmation_log']),
    }

    if agent['feedback_log']:
        domains = [r['correct'] for r in agent['feedback_log']]
        summary['corrections_per_domain'] = dict(Counter(domains))

    if agent['keyword_rules']:
        summary['rules_count'] = {d: len(kw) for d, kw in agent['keyword_rules'].items()}

    return summary

def load_agent_state(agent):
    stateDir = agent['state_dir']

    logPath = os.path.join(stateDir, 'feedback_log.csv')
    if os.path.exists(logPath):
        with open(logPath, 'r', encoding='utf-8') as f:
            agent['feedback_log'] = list(csv.DictReader(f))
        logger.info("Loaded %d feedback records", len(agent['feedback_log']))

    confPath = os.path.join(stateDir, 'confirmation_log.csv')
    if os.path.exists(confPath):
        with open(confPath, 'r', encoding='utf-8') as f:
            agent['confirmation_log'] = list(csv.DictReader(f))
        logger.info("Loaded %d confirmation records", len(agent['confirmation_log']))

    rulesPath = os.path.join(stateDir, 'keyword_rules.json')
    if os.path.exists(rulesPath):
        with open(rulesPath, 'r') as f:
            data = json.load(f)
        agent['keyword_rules'] = defaultdict(Counter, {d: Counter(kw) for d, kw in data.items()})
        logger.info("Keyword rules loaded for %d domains", len(agent['keyword_rules']))

def compute_log_odds(agent, text):
    kwRules  = agent['keyword_rules']
    labelEnc = agent['label_encoder']

    if not kwRules:
        return {}

    words         = set(text.lower().split())
    logOddsScores = {d: 0.0 for d in labelEnc.classes_}
    matchedKws    = {d: [] for d in labelEnc.classes_}

    for kw in words:
        freqs     = {d: kwRules[d].get(kw, 0) for d in labelEnc.classes_}
        totalFreq = sum(freqs.values())
        if totalFreq == 0:
            continue

        for domain in labelEnc.classes_:
            targetCount = freqs[domain]
            otherCount  = totalFreq - targetCount
            odds = (targetCount + 1) / (otherCount + 1)
            lo   = np.log(odds)

            if lo != 0:
                logOddsScores[domain] += lo
                matchedKws[domain].append(f"{kw}({lo:.2f})")

    print('\nKeyword Matches:')
    for domain in labelEnc.classes_:
        if matchedKws[domain]:
            print(f"    {domain}: {', '.join(matchedKws[domain])} = {logOddsScores[domain]:.2f} total")

    return logOddsScores


def update_keyword_rules(agent, text, correctDomain):
    wordFreq    = Counter(text.lower().split())
    topKeywords = [w for w, _ in wordFreq.most_common(5)]

    print(f"Top 5 keywords: {topKeywords}")

    for word in topKeywords:
        oldWeight = agent['keyword_rules'][correctDomain][word]
        agent['keyword_rules'][correctDomain][word] += 1
        print(f"    '{word}': {oldWeight} → {agent['keyword_rules'][correctDomain][word]}")

    logger.info("Updated rules for '%s' with keywords: %s", correctDomain, topKeywords)

def append_correction_for_training(stateDir, emailId, fullText, correctDomain):
    trainPath  = os.path.join(stateDir, 'corrections_training.csv')
    fileExists = os.path.exists(trainPath)
    with open(trainPath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['email_id', 'lemmas', 'trueDomain'])
        if not fileExists:
            writer.writeheader()
        writer.writerow({'email_id': emailId, 'lemmas': fullText, 'trueDomain': correctDomain})

def save_agent_state(agent):
    stateDir = agent['state_dir']

    logPath = os.path.join(stateDir, 'feedback_log.csv')
    with open(logPath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'email_id', 'predicted', 'correct', 'text_snippet'])
        writer.writeheader()
        writer.writerows(agent['feedback_log'])

    confPath = os.path.join(stateDir, 'confirmation_log.csv')
    with open(confPath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'email_id', 'predicted', 'correct',
            'text_snippet', 'type', 'original_confidence'
        ])
        writer.writeheader()
        writer.writerows(agent['confirmation_log'])

    rulesPath = os.path.join(stateDir, 'keyword_rules.json')
    serializable = {d: dict(kw) for d, kw in agent['keyword_rules'].items()}
    with open(rulesPath, 'w') as f:
        json.dump(serializable, f, indent=2)

    logger.info("Agent state saved to: %s", stateDir)