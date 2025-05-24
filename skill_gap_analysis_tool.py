import json
from sklearn.feature_extraction.text import TfidfVectorizer


# loading data
with open('intern_skills_data.json', 'r') as file:
    intern_skills_data = json.load(file)

with open('industry_job_description.json', 'r') as file:
    job_descriptions_data = json.load(file)

print("\nALL DATA")
print("\nInterns Data:\n", intern_skills_data)
print("\nJob Descriptions:\n", job_descriptions_data)

#  Skills ko Text Mein Convert Karna (String format)
# Machine Learning algorithms directly lists ya arrays ko samajh nahi sakte. 
# Humein intern aur job skills ko text (sentence-like format) mein convert karna hota hai 
# taake un par TF-IDF apply kiya ja sake.

intern_skills = [' '. join(intern['skills']) for intern in intern_skills_data]
job_descriptions = [' '.join(job['required_skills']) for job in job_descriptions_data]

print("\nConverted Skills and Job Descriptions:")
print("\nIntern Skills (text):", intern_skills)
print("\nJob Descriptions (text):", job_descriptions)

# TF-IDF Vectorizer Apply Karna
# TF-IDF (Term Frequency - Inverse Document Frequency) 
# ek algorithm hai jo text ko numbers (vectors) mein convert karta hai. 
# Har skill ka importance calculate karta hai.


# initializing TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# fit aur transform apply on Intern skills
intern_tfidf = vectorizer.fit_transform(intern_skills)

# same for job descriptions
job_tfidf = vectorizer.transform(job_descriptions)

# Output shape check karna
print("\nTF-IDF Shapes:")
print("\nIntern TF-IDF Shape:", intern_tfidf.shape)
print("\nJob TF-IDF Shape:", job_tfidf.shape)
