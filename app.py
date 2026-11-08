# -*- coding: cp1251 -*-
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from shap_file_stremlit import *
from train_functions import *
from config import inference_model_name
from streamlit_option_menu import option_menu
import io
import torch
from unittest.mock import patch


def load_model_on_cpu(path):
    def load_storage(data):
        return torch.load(
            io.BytesIO(data),
            map_location="cpu",
            weights_only=False,
        )

    with patch("torch.storage._load_from_bytes", load_storage):
        return joblib.load(path)


st.set_page_config(layout="wide")

st.markdown("""
    <style>
    /* Ñòèëü äëÿ âñåõ êíîïîê â ïðèëîæåíèè */
    div.stButton > button:first-child {
        background-color: #e52d27; /* Öâåò ôîíà */
        color: white;               # Öâåò òåêñòà
        border-radius: 10px;        /* Ñêðóãëåíèå óãëîâ */
        border: 2px solid #e52d27;  /* Ãðàíèöà */
        font-family: "Segoe UI", sans-serif;
        height: 3em;
        width: 100%;                /* Ðàñòÿíóòü íà âñþ øèðèíó êîëîíêè */
        transition: 0.3s;
    }

    /* Ýôôåêò ïðè íàâåäåíèè */
    div.stButton > button:first-child:hover {
        background-color: 	#bcbdbc;
        border-color: #bcbdbc;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Çíà÷åíèÿ äëÿ èíòåðôåéñà ------
marital_dict = {'íå â áðàêå': 'single', 'â áðàêå': 'married', 'â ðàçâîäå': 'divorced'}
job_dict = {
    'àäìèíèñòðàòîð': 'admin.', 'ðàáî÷èé': 'blue-collar',
    'ïðåäïðèíèìàòåëü': 'entrepreneur', 'äîìîõîçÿéêà': 'housemaid',
    'ìåíåäæìåíò / ðóêîâîäèòåëü': 'management', 'ïåíñèîíåð': 'retired',
    'ñàìîçàíÿòûé': 'self-employed', 'çàíÿò â ñôåðå óñëóã': 'services',
    'ñòóäåíò': 'student', 'òåõíè÷åñêèé ñïåöèàëèñò': 'technician',
    'áåçðàáîòíûé': 'unemployed', 'íåèçâåñòíî': 'unknown'}
education_dict = {'íà÷àëüíîå': 'primary', 'ñðåäíåå': 'secondary',
                  'âûñøåå': 'tertiary', 'íåèçâåñòíî': 'unknown'}
yes_or_no_dict = {'äà, åñòü': 'yes', 'íåò': 'no'}
contact_dict = {'ìîáèëüíûé òåëåôîí (ñìàðòôîí)': 'cellular', 'ñòàöèîíàðíûé òåëåôîí': 'telephone',
                'íåèçâåñòíî': 'unknown'}
poutcome_dict = {'óñïåõ': 'success', 'îòêàç': 'failure', 'äðóãîå': 'other', 'íåèçâåñòíî': 'unknown'}
months_dict = {'ÿíâàðü': 'jan', 'ôåâðàëü': 'feb', 'ìàðò': 'mar', 'àïðåëü': 'apr',
               'ìàé': 'may', 'èþíü': 'jun', 'èþëü': 'jul', 'àâãóñò': 'aug', 'ñåíòÿáðü': 'sep',
               'îêòÿáðü': 'oct', 'íîÿáðü': 'nov', 'äåêàáðü': 'dec'}
# ---------
num_of_models = 15


models = []

for model_type in ["TabM", "CatBoost", "LightGBM"]:
    for idx in range(1, 6):
        path = (
            f"models/{model_type}_"
            f"{inference_model_name}_fold_{idx}.pkl"
        )

        if model_type == "TabM":
            model = load_model_on_cpu(path)
            model.set_params(device="cpu")
        else:
            model = joblib.load(path)

        models.append(model)

meta_model = joblib.load(
    "models/meta_model_logistic_regression.pkl"
)


params = ['age', 'job', 'marital', 'education', 'default', 'balance',
          'housing', 'loan', 'contact', 'day', 'month', 'duration',
          'campaign', 'pdays', 'previous', 'poutcome']

numeric_features = ['age', 'balance', 'day', 'duration',
                    'campaign', 'pdays', 'previous']

categorical_features = ['job', 'marital', 'education', 'default',
                        'housing', 'loan', 'contact', 'month', 'poutcome']

st.write("")
col1, col2 = st.columns([20, 1])

with col2:
    st.image('image.svg', width=40)

selected = option_menu(
    menu_title='ÌÎÄÅËÜ ÄËß ÎÖÅÍÊÈ ÂÅÐÎßÒÍÎÑÒÈ ÏÎÄÏÈÑÀÍÈß ÄÎÃÎÂÎÐÀ ÍÀ ÎÒÊÐÛÒÈÅ ÄÅÏÎÇÈÒÀ',
    options=['Îäèíî÷íûé ïðîãíîç', 'Àíàëèòèêà', 'Ìíîæåñòâåííûé ïðîãíîç'],  # Íàçâàíèÿ âêëàäîê
    icons=["file-binary-fill", "bar-chart-fill", "filetype-csv"],  # Èêîíêè äëÿ âêëàäîê
    menu_icon="stack",  # Èêîíêà ñàìîãî ìåíþ
    default_index=0,  # Êàêàÿ âêëàäêà îòêðûòà ïî óìîë÷àíèþ
    orientation="horizontal",  # Äåëàåò ìåíþ ãîðèçîíòàëüíûì (êàê âêëàäêè)
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"font-size": "20px"},
        "menu-title": {
            "color": "#FF4B4B",
            "font-family": "Segoe UI",
            "font-weight": "bold",
            "font-size": "18px",

        },
        "nav-link": {
            "font-family": "Segoe UI",
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee"}})

if selected == 'Îäèíî÷íûé ïðîãíîç':
    st.write("Ââåäèòå ïàðàìåòðû êëèåíòà:")
    col_1, col_2 = st.columns(2)

    with col_1:
        st.divider()
        age = st.slider("Âîçðàñò (age)", min_value=18, max_value=95, value=35)
        st.divider()
        job_rus = st.selectbox("Òèï çàíÿòîñòè (job)",
                               ['àäìèíèñòðàòîð', 'ðàáî÷èé', 'ïðåäïðèíèìàòåëü', 'äîìîõîçÿéêà',
                                'ìåíåäæåð / ðóêîâîäèòåëü', 'ïåíñèîíåð', 'ñàìîçàíÿòûé', 'çàíÿò â ñôåðå óñëóã',
                                'ñòóäåíò', 'òåõíè÷åñêèé ñïåöèàëèñò',
                                'áåçðàáîòíûé', 'íåèçâåñòíî'])
        job = job_dict[job_rus]

        st.divider()
        marital_rus = st.selectbox("Ñåìåéíîå ïîëîæåíèå (marital)",
                                   ['íå â áðàêå', 'â áðàêå', 'â ðàçâîäå'])
        marital = marital_dict[marital_rus]
        st.divider()
        education_rus = st.selectbox("Óðîâåíü îáðàçîâàíèÿ (education)",
                                     ['íà÷àëüíîå', 'ñðåäíåå', 'âûñøåå', 'íåèçâåñòíî'])
        education = education_dict[education_rus]

        st.divider()
        default_rus = st.selectbox("Åñòü ëè ïðîñðî÷êà ïëàòåæà ïî êðåäèòó? (default)",
                                   ['äà, åñòü', 'íåò'])
        default = yes_or_no_dict[default_rus]

        st.divider()
        balance = st.slider("Ñðåäíåãîäîâîé áàëàíñ â ïåðåâîäå íà Åâðî (balance)", min_value=-8019, max_value=99717,
                            value=0, step=1)
        st.divider()
        housing_rus = st.selectbox("Åñòü ëè æèëèùíûé êðåäèò? (housing)",
                                   ['äà, åñòü', 'íåò'])
        housing = yes_or_no_dict[housing_rus]

        st.divider()

        loan_rus = st.selectbox("Åñòü ëè ïîòðåáèòåëüñêèé êðåäèò? (loan)",
                                ['äà, åñòü', 'íåò'])
        loan = yes_or_no_dict[loan_rus]

        st.divider()

    with col_2:

        st.divider()
        contact_rus = st.selectbox("Òèï êîíòàêòà (contact)",
                                   ['ìîáèëüíûé òåëåôîí (ñìàðòôîí)', 'ñòàöèîíàðíûé òåëåôîí', 'íåèçâåñòíî'])
        contact = contact_dict[contact_rus]

        st.divider()
        day = st.slider("Ïîñëåäíèé äåíü êîíòàêòà çà ìåñÿö (day)", 1, 31, 15)
        st.divider()
        month_rus = st.selectbox("Ïîñëåäíèé ìåñÿö êîíòàêòà çà ãîä (month)",
                                 ['ÿíâàðü', 'ôåâðàëü', 'ìàðò', 'àïðåëü', 'ìàé', 'èþíü',
                                  'èþëü', 'àâãóñò', 'ñåíòÿáðü', 'îêòÿáðü', 'íîÿáðü', 'äåêàáðü'])
        month = months_dict[month_rus]

        st.divider()
        duration = st.slider("Ïðîäîëæèòåëüíîñòü ïîñëåäíåãî êîíòàêòà â ñåêóíäàõ (duration)", min_value=1, max_value=4918,
                             value=1, step=1)
        st.divider()
        campaign = st.slider("Êîëè÷åñòâî êîíòàêòîâ, ñîâåðø¸ííûõ â õîäå êàìïàíèè (campaign)", 1, 63, 1)
        st.divider()
        pdays = st.slider("Êîëè÷åñòâî äíåé ñ ìîìåíòà ïîñëåäíåãî îáðàùåíèÿ (pdays)",
                          -1, 871, 0)
        st.divider()
        previous = st.slider("Êîëè÷åñòâî êîíòàêòîâ, ñîâåðø¸ííûõ äî ýòîé êàìïàíèè (previous)", 0, 200, 0)
        st.divider()
        poutcome_rus = st.selectbox("Ðåçóëüòàò ïðåäûäóùåé ìàðêåòèíãîâîé êàìïàíèè (poutcome)",
                                    ['óñïåõ', 'îòêàç', 'äðóãîå', 'íåèçâåñòíî'])
        poutcome = poutcome_dict[poutcome_rus]

        st.divider()

    if st.button("Ñäåëàòü ïðîãíîç"):
        features = np.array([[age, job, marital, education, default, balance,
                              housing, loan, contact, day, month, duration,
                              campaign, pdays, previous, poutcome]])

        df = pd.DataFrame([features[0]], columns=params)
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = create_new_features(df)

        # ïîäãîòîâêà äàííûõ
        df = preprocessing(df)

        new_features = df.values  # Äëÿ ìîäåëåé, êîòîðûå èñïîëüçóþò ñîçäàííûå ôè÷è âìåñòî features

        probs = {'TabM': [], 'CatBoost': [], 'LightGBM': []}

        for idx in range(num_of_models):
            new_features_copy = new_features.copy()
            if idx <= 4:
                probs['TabM'].append(models[idx].predict_proba(df)[0][1])
            elif 5 <= idx <= 9:
                probs['CatBoost'].append(models[idx].predict_proba(new_features)[0][1])
            else:
                cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
                for col in cat_cols:
                    df[col] = df[col].astype("category")
                probs['LightGBM'].append(models[idx].predict_proba(df)[0][1])

        probs = {'TabM': [np.mean(probs['TabM'])],
                 'CatBoost': [np.mean(probs['CatBoost'])],
                 'LightGBM': [np.mean(probs['LightGBM'])]}
        probs = pd.DataFrame(probs)

        prob = meta_model.predict_proba(probs)[0][1]

        if prob > 0.37:
            color1, color2 = '#32e111', '#2bc10f'
            text = 'Êëèåíò ñäåëàåò äåïîçèò'
        else:
            color1, color2 = '#e52d27', '#b31217'
            text = 'Êëèåíò íå ñäåëàåò äåïîçèò'

        st.markdown(f"""
            <style>
                    .result-card {{
                        animation: fadeInUp 0.6s ease-out;
                    }}

                    @keyframes fadeInUp {{
                        from {{
                            opacity: 0;
                            transform: translateY(20px);
                        }}
                        to {{
                            opacity: 1;
                            transform: translateY(0);
                        }}
                    }}
                    </style>
            <div style="
                background-color: #ffffff; 
                padding: 20px; 
                border-radius: 15px; 
                border: 1px solid #eee;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                text-align: center;">
                <h3 style="font-family: 'Segoe UI'; color: #31333F; margin: 0; font-size: 18px;
                background: -webkit-linear-gradient({color1}, {color2});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;">
                    {text}
                </h3>
                <p style="
                    font-family: 'Segoe UI'; 
                    font-size: 70px; 
                    font-weight: 800; 
                    background: -webkit-linear-gradient({color1}, {color2});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 10px 0;">
                    {str(round(prob * 100, 2)).replace('.', ',')}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        categorical_features = df.select_dtypes(
            include=["object", "string", "category"]).columns.tolist()

        params = list(df.columns)
        for idx in range(num_of_models):
            if 5 <= idx <= 9:
                html_file = save_force_plot(models[idx], df, categorical_features, params,
                                            f"shap_force_{idx-5}.html")
            else:
                pass

if selected == 'Àíàëèòèêà':
    for idx in range(5):
        st.write(f'Ðåøàþùèå ïðèçíàêè äëÿ {idx + 1}-é ìîäåëè')
        st.components.v1.html(
            open(f"shap_force_{idx}.html", 'r', encoding='utf-8').read(),
            height=170,
            scrolling=True)

if selected == 'Ìíîæåñòâåííûé ïðîãíîç':
    uploaded = st.file_uploader("Çàãðóçèòå äàííûå â âèäå csv-ôàéëà", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # ïîäãîòîâêà äàííûõ
        df = create_new_features(df)
        df = preprocessing(df)
        new_features = df.values

        probs = {'TabM': [], 'CatBoost': [], 'LightGBM': []}

        for idx in range(num_of_models):
            new_features_copy = new_features.copy()
            if idx <= 4:
                probs['TabM'].append(models[idx].predict_proba(df)[:, 1])
            elif 5 <= idx <= 9:
                probs['CatBoost'].append(models[idx].predict_proba(new_features)[:, 1])
            else:
                cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
                for col in cat_cols:
                    df[col] = df[col].astype("category")
                probs['LightGBM'].append(models[idx].predict_proba(df)[:, 1])

        tabm_mean = np.mean(probs['TabM'], axis=0) if probs['TabM'] else np.zeros(len(df))
        cat_mean = np.mean(probs['CatBoost'], axis=0) if probs['CatBoost'] else np.zeros(len(df))
        lgb_mean = np.mean(probs['LightGBM'], axis=0) if probs['LightGBM'] else np.zeros(len(df))
        matrix_data = np.column_stack([tabm_mean, cat_mean, lgb_mean])

        probs_df = pd.DataFrame(matrix_data, columns=['TabM', 'CatBoost', 'LightGBM'])

        prob = meta_model.predict_proba(probs_df.values)[:, 1]

        df["probability"] = prob
        df['result'] = (df['probability'] >= 0.37).astype(int)

        st.write(df[['probability', 'result']])

        df.to_csv(r"prediction_results.csv", index=False)
        st.info("Ðåçóëüòàòû ñîõðàíåíû â prediction_results.csv")
