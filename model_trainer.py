from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle

def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Skills'])

    model = KMeans(n_clusters=5, random_state=42)
    clusters = model.fit_predict(X)
    df['cluster'] = clusters

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return df

def classify_jobs(df):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform(df['Skills'])
    df['cluster'] = model.predict(X)
    return df


# main.py (or your main execution cell/script)
if __name__ == "__main__":
    # Step 1: scrape jobs
    df_jobs = scrape_karkidi_jobs(keyword="data science", pages=2)
    print("Scraped jobs:")
    print(df_jobs.head())

    # Step 2: train model on scraped data
    df_trained = train_model(df_jobs)
    print("
Jobs with clusters:")
    print(df_trained[['Title', 'Skills', 'cluster']].head())

    # Step 3: classify new jobs (simulate by scraping again)
    df_new_jobs = scrape_karkidi_jobs(keyword="data science", pages=1)  # new scrape
    df_classified = classify_jobs(df_new_jobs)
    print("
New jobs classified:")
    print(df_classified[['Title', 'Skills', 'cluster']].head())
