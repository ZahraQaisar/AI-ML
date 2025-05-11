import pandas as pd
from sklearn.decomposition import NMF

# Dataset from CSV file
df = pd.read_csv("intern_data.csv")  
print(df.head())  # Checking first 5 rows


#creating a ratings matrix table
ratings_matrix = df.pivot_table(
    index='intern_id',
    columns='course_id',
    values='rating',
    fill_value=0  # Replace NaN with 0 (those who haven't rated the course)
)
print(ratings_matrix)


# Non-Negative Matrix Factorization
# NMF model (Collaborative Filtering)
model = NMF(n_components=2)  # 2 hidden features - Step 1: 2 features (e.g., "Programming" aur "Maths")
model.fit(ratings_matrix)   # Step 2: Training model based on ratings data 
# n_components=2 means:
# The model will find 2 main groups in your data.
# Example: "Tech Courses" vs "Non-Tech Courses".
# If you use n_components=5:
# It will find 5 groups instead.
# Example: "Python", "Maths", "Design", etc.
# fit(ratings_matrix) means:
# You're telling the model: "Hey, learn patterns from this ratings table!"


intern_id = 1  # ID of the intern for whom we want recommendations
user_ratings = ratings_matrix.loc[intern_id].values.reshape(1, -1)
# This gets all the course ratings for intern #1 from the big ratings table
# Converts the ratings into a special format that our model can understand
# reshape(1, -1) means:
# Keep as 1 row (1 intern)
# All columns (all courses)

# Predicted ratings for all courses
# Calculates how much they would rate each course based on those interests
# Works like: (interest in tech × tech-score of course) + (interest in design × design-score of course)
predicted_ratings = model.inverse_transform(model.transform(user_ratings))

# Top 3 courses suggestions
recommendations = pd.DataFrame({
    'course_id': ratings_matrix.columns,
    'predicted_rating': predicted_ratings[0]
}).sort_values('predicted_rating', ascending=False).head(3)


print("Recommendations for Intern", intern_id)
print(recommendations)