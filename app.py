import streamlit as st

st.set_page_config(layout="wide")
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from collections import Counter

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

df = pd.read_csv("data/Tweets.csv")


# -------------------------------
# CLEAN FUNCTION
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    stop_words = set(stopwords.words("english"))
    text = " ".join([w for w in text.split() if w not in stop_words])

    return text


df["clean_text"] = df["text"].apply(clean_text)

# -------------------------------
# CONTENT CLASSIFICATION
# -------------------------------
complaint_keywords = ["delay", "cancelled", "late", "bad", "worst"]
service_keywords = ["service", "staff", "support"]
positive_keywords = ["good", "great", "love", "excellent"]


def classify_content(text):
    if any(w in text for w in complaint_keywords):
        return "Complaint"
    elif any(w in text for w in service_keywords):
        return "Service"
    elif any(w in text for w in positive_keywords):
        return "Praise"
    else:
        return "Other"


df["content_type"] = df["clean_text"].apply(classify_content)

# -------------------------------
# UI START
# -------------------------------
st.title("✈️ Twitter Airline Sentiment Dashboard")

# Sidebar
airline = st.sidebar.selectbox("Select Airline", df["airline"].unique())
filtered_df = df[df["airline"] == airline].copy()

# ===============================
# SECTION 1: OVERVIEW
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="airline_sentiment", data=filtered_df, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("Content Type Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="content_type", data=filtered_df, ax=ax2)
    st.pyplot(fig2)

# ===============================
# SECTION 2: COMPETITOR ANALYSIS
# ===============================
sentiment_ratio = df.groupby(["airline", "airline_sentiment"]).size().unstack()
sentiment_ratio = sentiment_ratio.div(sentiment_ratio.sum(axis=1), axis=0)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Competitor Comparison")
    fig3, ax3 = plt.subplots()
    sentiment_ratio.plot(kind="bar", stacked=True, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

with col4:
    st.subheader("Heatmap")
    fig4, ax4 = plt.subplots()
    sns.heatmap(sentiment_ratio, annot=True, ax=ax4)
    st.pyplot(fig4)

# ===============================
# SECTION 3: WORDCLOUD
# ===============================
st.subheader("WordCloud")

# TOP WORDS (instead of wordcloud)
words = " ".join(filtered_df["clean_text"]).split()
freq = Counter(words).most_common(10)

w = [i[0] for i in freq]
c = [i[1] for i in freq]

fig5, ax5 = plt.subplots()
ax5.bar(w, c)
plt.xticks(rotation=45)
st.pyplot(fig5)

# ===============================
# SECTION 4: ADVANCED ANALYSIS
# ===============================
filtered_df["tweet_length"] = filtered_df["clean_text"].apply(len)

col5, col6 = st.columns(2)

with col5:
    st.subheader("Tweet Length vs Sentiment")
    fig6, ax6 = plt.subplots()
    sns.boxplot(x="airline_sentiment", y="tweet_length", data=filtered_df, ax=ax6)
    st.pyplot(fig6)

with col6:
    st.subheader("Top Words")
    words = " ".join(filtered_df["clean_text"]).split()
    freq = Counter(words).most_common(10)

    w = [i[0] for i in freq]
    c = [i[1] for i in freq]

    fig7, ax7 = plt.subplots()
    ax7.bar(w, c)
    plt.xticks(rotation=45)
    st.pyplot(fig7)

# ===============================
# SECTION 5: SAMPLE DATA
# ===============================
st.subheader("Sample Tweets")
st.write(filtered_df[["text", "airline_sentiment"]].head(5))


# ===============================
# SECTION 6: CONTENT GENERATOR
# ===============================
def generate_marketing_content(airline):
    return f"""
    ✈️ Fly with {airline}!
    Enjoy comfort, reliability and excellent service.
    Book your journey today!
    """


st.subheader("Marketing Content Generator")

airline_name = st.text_input("Enter Airline Name")

if st.button("Generate Content"):
    st.success(generate_marketing_content(airline_name))

# ===============================
# SECTION 7: SENTIMENT PREDICTION
# ===============================
st.subheader("Sentiment Prediction")

user_input = st.text_area("Enter a tweet")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    st.success(f"Predicted Sentiment: {pred[0]}")
