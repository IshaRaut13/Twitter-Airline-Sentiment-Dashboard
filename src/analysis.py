import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

from preprocess import load_and_clean_data

df = load_and_clean_data("../data/Tweets.csv")

# -------------------------------
# EXP 4: Sentiment Distribution
# -------------------------------
sns.countplot(x="airline_sentiment", data=df)
plt.title("Sentiment Distribution")
plt.show()

# -------------------------------
# EXP 4: Airline vs Sentiment
# -------------------------------
sns.countplot(x="airline", hue="airline_sentiment", data=df)
plt.xticks(rotation=45)
plt.title("Airline vs Sentiment")
plt.show()

# -------------------------------
# EXP 9: Competitor Analysis
# -------------------------------
sentiment_ratio = df.groupby(["airline", "airline_sentiment"]).size().unstack()
sentiment_ratio = sentiment_ratio.div(sentiment_ratio.sum(axis=1), axis=0)

sentiment_ratio.plot(kind="bar", stacked=True)
plt.title("Competitor Sentiment Comparison")
plt.show()

# Heatmap
sns.heatmap(sentiment_ratio, annot=True)
plt.title("Sentiment Heatmap")
plt.show()

# -------------------------------
# EXP 5: Content Analysis
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

sns.countplot(x="content_type", data=df)
plt.title("Content Type Distribution")
plt.show()

# -------------------------------
# WordCloud
# -------------------------------
text = " ".join(df["clean_text"])
wc = WordCloud(width=800, height=400).generate(text)

plt.imshow(wc)
plt.axis("off")
plt.title("WordCloud")
plt.show()

# -------------------------------
# Tweet Length
# -------------------------------
df["tweet_length"] = df["clean_text"].apply(len)

sns.boxplot(x="airline_sentiment", y="tweet_length", data=df)
plt.title("Tweet Length vs Sentiment")
plt.show()

# -------------------------------
# Top Words
# -------------------------------
words = " ".join(df["clean_text"]).split()
freq = Counter(words).most_common(10)

w = [i[0] for i in freq]
c = [i[1] for i in freq]

plt.bar(w, c)
plt.xticks(rotation=45)
plt.title("Top Words")
plt.show()

# -------------------------------
# EXP 6: Structure Model
# -------------------------------
structure = df.groupby(["airline", "airline_sentiment", "content_type"]).size()
print("\nStructure Model:\n", structure)
