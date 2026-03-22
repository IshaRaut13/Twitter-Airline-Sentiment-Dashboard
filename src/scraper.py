import pandas as pd


def simulate_scraping():
    df = pd.read_csv("../data/Tweets.csv")

    sample = df.sample(5)

    print("Simulated Scraped Tweets:\n")
    print(sample[["airline", "text"]])


if __name__ == "__main__":
    simulate_scraping()
