

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import joblib
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# -----------------------
# Streamlit setup
# -----------------------
st.set_page_config(layout="wide", page_title="Sentiment Analysis Dashboard")

# -----------------------
# Utility functions
# -----------------------
@st.cache_data
def clean_text_series(series: pd.Series) -> pd.Series:
    def _clean(t):
        if pd.isna(t):
            return ""
        t = str(t).lower()
        t = re.sub(r"http\S+|www\S+|https\S+", " ", t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    return series.astype(str).map(_clean)

def load_models(models_dir="./models"):
    models = {}
    p = Path(models_dir)
    expected = {
        "NaiveBayes": "naivebayes_model.pkl",
        "LogisticRegression": "logisticregression_model.pkl",
        "KNN": "knn_model.pkl"
    }
    alt = {
        "NaiveBayes": "nb_model.pkl",
        "LogisticRegression": "logisticregression_model.pkl",
        "KNN": "knn_model.pkl"
    }
    for name in expected:
        if (p / expected[name]).exists():
            models[name] = joblib.load(p / expected[name])
        elif (p / alt[name]).exists():
            models[name] = joblib.load(p / alt[name])
    return models

def build_pipelines():
    tfidf = ("tfidf", TfidfVectorizer(ngram_range=(1,1), max_features=3000))
    pipelines = {
        'NaiveBayes': Pipeline([ tfidf, ('clf', MultinomialNB()) ]),
        'LogisticRegression': Pipeline([ tfidf, ('clf', LogisticRegression(max_iter=1000, random_state=42)) ]),
        'KNN': Pipeline([tfidf, ('clf', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])
        # 'RandomForest': Pipeline([ tfidf, ('clf', RandomForestClassifier(n_estimators=200, random_state=42)) ])
    }
    return pipelines

def train_models(df, test_size=0.2, random_state=42):
    # if len(df) > sample_size:
    #     st.warning(f"Dataset has {len(df):,} samples — using a subset of {sample_size:,} for faster training.")
    #     df = df.sample(sample_size, random_state=random_state)

    X = df['clean_review']
    y = df['label']
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipelines = build_pipelines()
    trained = {}
    evals = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred, digits=4, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=['negative','neutral','positive'])
        evals[name] = {"accuracy": acc, "report": cr, "confusion_matrix": cm}
    return trained, evals

def predict_with_models(models, texts):
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(texts)
    return preds

def majority_vote(preds_dict):
    dfp = pd.DataFrame(preds_dict)
    def majority(row):
        counts = row.value_counts()
        if counts.empty:
            return "neutral"
        top = counts[counts == counts.max()].index.tolist()
        if len(top) == 1:
            return top[0]
        if 'neutral' in top:
            return 'neutral'
        return sorted(top)[0]
    return dfp.apply(majority, axis=1).values

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    labels = ['negative','neutral','positive']
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.tight_layout()
    return fig

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.title("Controls")


csv_path = st.sidebar.text_input("Enter CSV file path", value="amazon_beauty_reviews_dataset.csv")
models_dir = st.sidebar.text_input("Models directory", value="./models")
st.sidebar.markdown("---")

models_loaded = load_models(models_dir)
st.sidebar.markdown(f"Models found: {', '.join(models_loaded.keys()) if models_loaded else 'None'}")
st.sidebar.markdown("Train new models below if needed.")
train_from_upload = st.sidebar.button("Train models from this dataset")

# -----------------------
# Load dataset
# -----------------------
st.title("Sentiment Analysis Dashboard")

if not os.path.exists(csv_path):
    st.error(f"CSV file not found: {csv_path}")
    st.stop()

# Attempt multiple encodings
encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1']
df = None
for enc in encodings_to_try:
    try:
        df = pd.read_csv(csv_path, encoding=enc)
        st.success(f"Loaded CSV using encoding: {enc}")
        break
    except Exception:
        continue

if df is None:
    st.error("Failed to load CSV. Please check the file encoding.")
    st.stop()

if 'review' not in df.columns:
    st.error("Dataset must have a 'review' column.")
    st.stop()

df['clean_review'] = clean_text_series(df['review'])
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
else:
    df['date'] = pd.NaT

# -----------------------
# Map numeric labels 1-5 to sentiment
# -----------------------
if 'label' in df.columns:
    df['label'] = df['label'].apply(lambda x: int(x) if pd.notnull(x) else 3)  # default 3 = neutral
    df['label'] = df['label'].map({
        1: 'negative',
        2: 'negative',
        3: 'neutral',
        4: 'positive',
        5: 'positive'
    })
else:
    df['label'] = None

st.subheader("Dataset preview")
st.dataframe(df.head(50))

# -----------------------
# Model training / loading
# -----------------------
trained_models = {}
evals = {}

if train_from_upload:
    if df['label'].isnull().all():
        st.error("Training requires a 'label' column with positive/neutral/negative values.")
    else:
        with st.spinner("Training models..."):
            trained_models, evals = train_models(df[df['label'].notnull()].reset_index(drop=True))
            os.makedirs(models_dir, exist_ok=True)
            for name, model in trained_models.items():
                joblib.dump(model, os.path.join(models_dir, f"{name.lower()}_model.pkl"))
            st.success(f"Trained and saved models to {models_dir}")

if not trained_models:
    trained_models = models_loaded

if not trained_models:
    st.warning("No models available. Either train new ones or place pre-trained models in ./models/.")
    st.stop()

# -----------------------
# Single review prediction
# -----------------------
st.subheader("Predict sentiment for a single review")

# User can pick a model for single-review prediction
single_model_options = list(trained_models.keys()) + ["Ensemble (majority)"]
selected_model_single = st.selectbox(
    "Choose model for single review prediction",
    options=single_model_options
)

# Text input for the review
single_review = st.text_input("Enter a review to predict its sentiment:")

if single_review:
    # Clean the input review
    single_clean = clean_text_series(pd.Series([single_review]))[0]

    # Predict using all trained models
    single_preds = predict_with_models(trained_models, [single_clean])

    # Ensemble prediction
    ensemble_pred = majority_vote(single_preds)

    # Pick the prediction based on selected model
    if selected_model_single == "Ensemble (majority)":
        final_pred = ensemble_pred[0]
    else:
        final_pred = single_preds[selected_model_single][0]

    # Display results
    st.write("**Predictions:**")
    for model_name, pred in single_preds.items():
        st.write(f"{model_name}: {pred[0]}")
    st.write(f"**Selected model prediction ({selected_model_single}): {final_pred}**")

# -----------------------
# Prediction Section
# -----------------------
st.subheader("Predict sentiment for prediction dataset")
model_options = list(trained_models.keys()) + ["Ensemble (majority)"]
selected_model = st.selectbox("Choose model to use for predictions", options=model_options)

if st.button("Run predictions"):
    texts = df['clean_review'].fillna("").values
    preds = predict_with_models(trained_models, texts)
    results_df = df.copy()
    for name, arr in preds.items():
        results_df[f"pred_{name}"] = arr
    results_df['pred_ensemble'] = majority_vote({k: v for k,v in preds.items()})
    final_col = 'pred_ensemble' if selected_model == "Ensemble (majority)" else f"pred_{selected_model}"
    results_df['predicted_label'] = results_df[final_col]

    st.success("Predictions complete!")

    st.subheader("Prediction distribution")
    dist = results_df['predicted_label'].value_counts().reindex(['positive','neutral','negative']).fillna(0)
    st.bar_chart(dist)

    st.subheader("Monthly sentiment trend (by predicted label)")
    if results_df['date'].notnull().any():
        results_df['month'] = results_df['date'].dt.to_period('M').dt.to_timestamp()
        counts = results_df.groupby(['month','predicted_label']).size().unstack(fill_value=0)
        st.line_chart(counts)
    else:
        st.info("No date column available for trend plotting.")

    st.subheader("Sample reviews by predicted sentiment")
    cols = st.columns(3)
    for i, lab in enumerate(['positive','neutral','negative']):
        with cols[i]:
            st.write(f"### {lab.capitalize()}")
            samples = results_df[results_df['predicted_label']==lab]['review'].head(6).tolist()
            if not samples:
                st.write("— none —")
            for s in samples:
                st.write("- " + str(s))

    if results_df['label'].notnull().any():
        st.subheader("Evaluation (on labeled data)")
        eval_choice = st.selectbox("Evaluate which model?", options=model_options)
        y_pred = results_df['pred_ensemble'] if eval_choice == "Ensemble (majority)" else results_df[f"pred_{eval_choice}"]
        y_true = results_df['label']
        acc = accuracy_score(y_true, y_pred)
        st.write(f"Accuracy: {acc:.4f}")
        st.text("Classification report:")
        st.text(classification_report(y_true, y_pred, digits=4))
        cm = confusion_matrix(y_true, y_pred, labels=['negative','neutral','positive'])
        fig = plot_confusion_matrix(cm, title=f"Confusion Matrix — {eval_choice}")
        st.pyplot(fig)

    csv_buf = io.StringIO()
    results_df.to_csv(csv_buf, index=False)
    st.download_button("Download predictions CSV", data=csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")

    if evals:
        st.subheader("Training-time evaluation summary")
        rows=[]
        for m,info in evals.items():
            rows.append({"model": m, "accuracy": info['accuracy']})
        eval_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
        st.table(eval_df)

st.sidebar.markdown("---")
