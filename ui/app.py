import os
import sys
import subprocess
import streamlit as st
import numpy as np
import pandas as pd
import joblib

root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_directory)

from ui.feedback_agent import make_agent, classify_email, confirm_prediction, log_feedback, get_feedback_summary, load_agent_state

output_directory = os.path.join(root_directory, 'models')
modelp = os.path.join(output_directory, 'svm_model.joblib')
tfidf_ldap = os.path.join(output_directory, 'tfidf_lda.joblib')
labelp = os.path.join(output_directory, 'label_encoder.joblib')
scalerp = os.path.join(output_directory, 'scaler.joblib')
agentstate_directory = os.path.join(root_directory, 'ui', 'agent_state')

domains = {
    '01_PC': '📋 Placement & Career',
    '02_AC': '📚 Academic Coursework',
    '03_ER': '📝 Exams and Records',
    '04_CS': '🚌 Campus Services',
    '05_EC': '🎭 Extracurriculars',
    '06_UA': '🏛️ University Administration',
    '07_PP': '👥 Peer to Peer',
    '08_NP': '🚫 Noise',
}

def load_artifacts():
    for name, path in [('Model', modelp), ('TF-IDF/LDA', tfidf_ldap),
                       ('Label Encoder', labelp), ('Scaler', scalerp)]:

        model = joblib.load(modelp)
        tfidf_lda = joblib.load(tfidf_ldap)
        label_enc = joblib.load(labelp)
        scaler = joblib.load(scalerp)

    return model, tfidf_lda, label_enc, scaler


def c_preprocess(subject, body):
    raw = (subject + ' ' + body).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')

    preprocess_directory = os.path.join(root_directory, 'c_engine')
    exe_path    = os.path.join(preprocess_directory, 'preprocess.exe')
    data_directory = os.path.join(root_directory, 'data')
    temp_in   = os.path.join(data_directory, 'ui_input.tsv')
    temp_out  = os.path.join(data_directory, 'ui_output.csv')

    with open(temp_in, 'w', encoding='utf-8') as f:
        f.write(f"UI_SINGLE\tUNKNOWN\t{raw}\n")

    subprocess.run([exe_path, temp_in, temp_out], cwd=preprocess_directory, capture_output=True, text=True)

    lemmas = raw.lower()
    try:
        if os.path.exists(temp_out):
            df = pd.read_csv(temp_out)
            if len(df) > 0:
                lemmas = str(df.iloc[0].get('lemmas', raw.lower()))
    except Exception:
        pass

    for p in [temp_in, temp_out]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    return lemmas


def extract_features(text, tfidf_lda, scaler):
    tfidf = tfidf_lda['tfidf']
    lda   = tfidf_lda['lda']

    tfidfMat = tfidf.transform([text])
    ldaVecs  = lda.transform(tfidfMat)

    X = np.hstack([tfidfMat.toarray(), ldaVecs])
    X = scaler.transform(X)
    return X


def main():
    st.set_page_config(page_title='Email Domain Classifier', layout='wide')

    st.title('Email Domain Classifier')
    st.markdown('Classify university emails into domain categories')

    model, tfidf_lda, label_enc, scaler = load_artifacts()

    if 'agent' not in st.session_state:
        agent = make_agent(model=model, label_encoder=label_enc, state_dir=agentstate_directory)
        load_agent_state(agent)
        st.session_state.agent = agent

    agent = st.session_state.agent

    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'clean_text' not in st.session_state:
        st.session_state.clean_text = ''

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Email Input')
        subject = st.text_input('Subject', placeholder='Enter the subject of the email', key='subject_input')
        body    = st.text_area('Email Body', height=200, placeholder='Enter the body of the email', key='body_input')

    with col2:
        st.subheader('Domain Categories')
        for code, desc in domains.items():
            st.markdown(f"- **{code}**: {desc}")

    if st.button('Classify Email', type='primary', use_container_width=True):
        with st.spinner('Loading...'):
            clean_text = c_preprocess(subject, body)
            features   = extract_features(clean_text, tfidf_lda, scaler)
            result     = classify_email(agent, clean_text, features)

            st.session_state.result     = result
            st.session_state.clean_text = clean_text

    result = st.session_state.result
    print(result)
    if result:
        st.markdown('### Classification Result')
        col_r1, col_r2, col_r3 = st.columns(3)

        domain = result['predicted_domain']
        conf   = result['confidence']
        review = result['needs_review']

        with col_r1:
            st.metric('Predicted Domain', domains.get(domain, domain))
        with col_r2:
            st.metric('Confidence', f"{conf:.1%}")
        with col_r3:
            if review:
                st.error('Manual Review')
            else:
                st.success('High Confidence')

        if result['source'] == 'keyword_rule':
            st.info(f"Used keyword rules (SVM predicted: {result['svm_prediction']})")

        st.subheader('Feedback give pls')

        feedback_type = st.radio(
            'Is the predicted domain correct?',
            options=['Yes', 'No'],
            index=0, horizontal=True, key='feedback_radio'
        )

        if feedback_type == 'No':
            correct_domain = st.selectbox(
                'Select the true domain:',
                options=list(domains.keys()),
                format_func=lambda x: f"{x} — {domains.get(x, x)}",
                key='feedback_select'
            )
        else:
            correct_domain = domain

        if st.button('Submit Feedback', key='feedback_btn'):
            email_id   = f"UI_{abs(hash(st.session_state.get('body_input', '')))}"
            clean_text = st.session_state.clean_text

            if feedback_type == 'Yes':
                st.success('Thank You')
                confirm_prediction(agent, email_id, clean_text, domain, conf)
            else:
                st.warning(f"Correction logged: {domain} - {correct_domain}")
                needs_retrain = log_feedback(agent, email_id, clean_text, domain, correct_domain)
                if needs_retrain:
                    st.info('Retraining needed (10 corrections)')

    with st.sidebar:
        st.header('Agent Status')
        summary = get_feedback_summary(agent)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric('Corrections', summary.get('total_corrections', 0))
        with col_s2:
            st.metric('Confirmations', summary.get('total_confirmations', 0))

        if summary.get('corrections_per_domain'):
            st.markdown('**Corrections by domain:**')
            for d, c in summary['corrections_per_domain'].items():
                st.text(f"  {d}: {c}")

        if summary.get('rules_count'):
            st.markdown('**Keyword rules:**')
            for d, c in summary['rules_count'].items():
                st.text(f"  {d}: {c} keywords")

if __name__ == '__main__':
    main()
