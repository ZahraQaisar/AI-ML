import json
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans


# Load intern skills data
with open('intern_skills_data.json', 'r') as file:
    intern_skills_data = json.load(file)

# Load industry job descriptions data
with open('industry_job_description.json', 'r') as file:
    job_descriptions_data = json.load(file)

# Printing to check if data loaded correctly
# print(intern_skills_data)
# print(job_descriptions_data)

from sklearn.feature_extraction.text import TfidfVectorizer

# Combine all intern skills into one string for TF-IDF processing
intern_skills = [' '.join(intern['skills']) for intern in intern_skills_data]

# Combine all job descriptions into one string for TF-IDF processing
job_descriptions = [' '.join(job['required_skills']) for job in job_descriptions_data]

# Initialize TF-IDF Vectorizer
# text data ko numeric vectors mein convertion
vectorizer = TfidfVectorizer()

# Fit and transform intern skills data
intern_tfidf = vectorizer.fit_transform(intern_skills)

# Transform job descriptions data
job_tfidf = vectorizer.transform(job_descriptions)

# Check the shape of the resulting TF-IDF matrices
print("Intern TF-IDF Shape: ", intern_tfidf.shape)
print("Job TF-IDF Shape: ", job_tfidf.shape)



# Define number of clusters (adjust this number based on your data)
num_clusters = 5

# Combine intern and job description TF-IDF matrices for clustering
all_tfidf = intern_tfidf.toarray().tolist() + job_tfidf.toarray().tolist()

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(all_tfidf)

# Get cluster labels
labels = kmeans.labels_

# Separate labels for interns and jobs
intern_labels = labels[:len(intern_skills)]  # Labels for interns
job_labels = labels[len(intern_skills):]  # Labels for job descriptions

# Print the cluster assignments
print("Intern Labels: ", intern_labels)
print("Job Labels: ", job_labels)

# For each intern, compare their skills with the job descriptions in the same cluster
for i, intern in enumerate(intern_skills_data):
    # Get the intern's cluster label
    intern_cluster = intern_labels[i]
    
    print(f"Intern {intern['name']} (Cluster {intern_cluster}):")
    
    # Get the job descriptions in the same cluster
    similar_jobs = [job['required_skills'] for j, job in enumerate(job_descriptions_data) if job_labels[j] == intern_cluster]
    
    # Identify skill gaps
    missing_skills = []
    for job_skills in similar_jobs:
        missing_skills += [skill for skill in job_skills if skill not in intern['skills']]
    
    # Print missing skills
    print(f"Missing Skills for Intern: {list(set(missing_skills))}\n")


# Example of suggesting training for missing skills
missing_skills = list(set(missing_skills))
for skill in missing_skills:
    print(f"Suggested Training for: {skill} - [Insert training link or course details]")

# Clustering (Group Banana) using K-Means
# K-Means algorithm similar items ko group (cluster) karta hai. 
# Hum interns aur job descriptions ko unke skill vectors ke basis par cluster karenge.


# defining number of clusters 
num_clusters = 3

# combining intern and job TF-IDF (in array format) for clustering(grouping according to skills)
all_tfidf = intern_tfidf.toarray().tolist() + job_tfidf.toarray().tolist()

# appyling K-Means clustering
# fit - training model on data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(all_tfidf)

# Cluster labels storing in a variable
labels = kmeans.labels_

# separate labels of interns and jobs
intern_labels = labels[:len(intern_skills)]
job_labels = labels[len(intern_skills):]

# Print labels
print("\nCluster Labels:")
print("\nIntern Labels:", intern_labels)
print("\nJob Labels:", job_labels)

# skill gap analysis for each intern
for i, intern in enumerate(intern_skills_data):
    # Intern cluster label
    intern_cluster = intern_labels[i]
    
    print(f"\nIntern {intern['name']} (Cluster {intern_cluster}):")
    
    # find job descriptions in that cluster
    # finding similar jobs in the same cluster
    similar_jobs = [
        job['required_skills']
        for j, job in enumerate(job_descriptions_data)
        if job_labels[j] == intern_cluster
    ]
    
    # collecting missing skills
    missing_skills = []
    for job_skills in similar_jobs:
        for skill in job_skills:
            if skill not in intern['skills']:
                missing_skills.append(skill)
    
    # prining unique missing skills
    print("Missing Skills for Intern:", list(set(missing_skills)))


# Final list of missing skills from last intern (optional: collect from all interns instead)
missing_skills = list(set(missing_skills))  # Unique list

# printing Training suggestions
print("\n🔧 Suggested Trainings:")
for skill in missing_skills:
    print(f"• Suggested Training for: {skill} - [Insert training link or course]")
