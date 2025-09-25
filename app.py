# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud
import io

# Load model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.set_page_config(layout="wide", page_title="Amazon Review Sentiment Demo")

st.title("Amazon Review Sentiment — Demo")
st.markdown("Local demo: paste a review or upload a CSV (with column `reviewText`)")

# Sidebar options
st.sidebar.header("Options")
show_wordcloud = st.sidebar.checkbox("Show Word Clouds", value=True)
show_confusion = st.sidebar.checkbox("Show Confusion Matrix (if available)", value=True)
top_n = st.sidebar.slider("Top N words (explainability)", 5, 30, 10)

# Single review prediction
st.subheader("Single review")
review_input = st.text_area("Paste a review here", height=120)
if st.button("Predict review sentiment"):
    if not review_input.strip():
        st.warning("Please paste a review")
    else:
        cleaned = review_input  # optionally add cleaning function
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0].max()
        label = "Positive" if int(pred)==1 else "Negative"
        st.success(f"Prediction: **{label}** (confidence {prob:.2f})")

# CSV upload bulk prediction
st.subheader("Bulk: upload CSV")
uploaded_file = st.file_uploader("Upload CSV with column `reviewText`", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'reviewText' not in df.columns:
        st.error("CSV must have a column named `reviewText`")
    else:
        st.info(f"Running predictions on {len(df)} rows...")
        X = vectorizer.transform(df['reviewText'].astype(str).tolist())
        preds = model.predict(X)
        df['pred_label'] = preds
        df['pred_label_str'] = df['pred_label'].map({1:"Positive", 0:"Negative"})
        st.write(df[['reviewText','pred_label_str']].head(20))

        # Download predictions
        towrite = io.BytesIO()
        df.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button(label="Download predictions CSV", data=towrite, file_name="predictions.csv", mime="text/csv")

        # Show distribution
        st.subheader("Predicted distribution")
        counts = df['pred_label_str'].value_counts()
        st.bar_chart(counts)

        # Wordclouds
        if show_wordcloud:
            pos_text = " ".join(df[df['pred_label']==1]['reviewText'].astype(str).tolist())
            neg_text = " ".join(df[df['pred_label']==0]['reviewText'].astype(str).tolist())

            col1,col2 = st.columns(2)
            with col1:
                st.write("Positive word cloud")
                if pos_text.strip():
                    wc = WordCloud(width=600, height=300).generate(pos_text)
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            with col2:
                st.write("Negative word cloud")
                if neg_text.strip():
                    wc = WordCloud(width=600, height=300).generate(neg_text)
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

        # Confusion matrix if test labels present in uploaded file
        if show_confusion and 'sentiment' in df.columns:
            y_true = df['sentiment'].astype(int)
            y_pred = df['pred_label'].astype(int)
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative","Positive"])
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            st.pyplot(fig)

        # Show top features (explainability)
        st.subheader("Model explainability — top words")
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        feat_df = pd.DataFrame({"feature":feature_names, "coef":coefs})
        top_pos = feat_df.sort_values("coef", ascending=False).head(top_n)
        top_neg = feat_df.sort_values("coef").head(top_n)
        col1,col2 = st.columns(2)
        with col1:
            st.write("Top positive words")
            st.table(top_pos.reset_index(drop=True))
        with col2:
            st.write("Top negative words")
            st.table(top_neg.reset_index(drop=True))

# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("Amazon Product Review Sentiment Analysis")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check for required columns
    required_columns = ["reviewText", "sentiment", "product_name"]
    if all(col in df.columns for col in required_columns):
        st.success("CSV loaded successfully!")

        # Show first few rows
        st.subheader("Preview of data")
        st.dataframe(df.head())

        # Sentiment Prediction
        st.subheader("Sentiment Analysis")
        df['predicted_sentiment'] = model.predict(vectorizer.transform(df['reviewText']))

        # Show confusion matrix
        cm = confusion_matrix(df['sentiment'], df['predicted_sentiment'])
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=ax)
        st.pyplot(fig)

        # Sentiment stats per product
        st.subheader("Sentiment per Product")
        sentiment_stats = df.groupby("product_name")['predicted_sentiment'].value_counts().unstack().fillna(0)
        st.dataframe(sentiment_stats)

    else:
        st.error(f"CSV must contain columns: {', '.join(required_columns)}")

