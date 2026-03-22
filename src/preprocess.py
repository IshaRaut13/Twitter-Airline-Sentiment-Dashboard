import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


def load_and_clean_data(path):
    df = pd.read_csv(path)

    df = df[["airline", "text", "airline_sentiment", "tweet_created"]]

    df["clean_text"] = df["text"].apply(clean_text)

    stop_words = set(stopwords.words("english"))
    df["clean_text"] = df["clean_text"].apply(
        lambda x: " ".join([word for word in x.split() if word not in stop_words])
    )

    return df
