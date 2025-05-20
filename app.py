import streamlit as st
import pandas as pd
import pickle

# Load your model and vectorizer saved from previous steps
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to classify jobs with model
def classify_jobs(df):
    X = vectorizer.transform(df['Skills'])
    df['cluster'] = model.predict(X)
    return df

# You can reuse your scraper here or load a sample CSV/DF for demo
def load_sample_jobs():
    # For demo, add some sample jobs or load from previous scrape
    data = {
        'Title': ['Data Scientist', 'Backend Engineer', 'ML Engineer'],
        'Skills': ['python, machine learning, data analysis', 
                   'java, api development, backend', 
                   'tensorflow, deep learning, python']
    }
    return pd.DataFrame(data)

st.title("Job Monitor and Cluster")

# Load jobs - in real app you can scrape here or upload CSV
df_jobs = load_sample_jobs()
df_jobs = classify_jobs(df_jobs)

skill_input = st.text_input("Enter your skills (comma separated):")

if skill_input:
    user_skills = skill_input.lower().split(',')
    user_skills = [s.strip() for s in user_skills]

    # Simple matching: find jobs with overlapping skills keywords
    def match_skills(skills_text):
        return any(skill in skills_text for skill in user_skills)

    matched_jobs = df_jobs[df_jobs['Skills'].apply(match_skills)]

    st.write(f"Jobs matching your skills ({', '.join(user_skills)}):")
    st.dataframe(matched_jobs[['Title', 'Skills', 'cluster']])
else:
    st.write("Enter skills to see matching jobs.")