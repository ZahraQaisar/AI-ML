import pandas as pd #to read the data/handle the data
from sklearn.model_selection import train_test_split #to split the data into training and testing sets
from sklearn.feature_extraction.text import CountVectorizer #to convert text data into numerical data
from sklearn.linear_model import LogisticRegression #to create machine learning model
from sklearn.metrics import accuracy_score #to evaluate the model/accuracy of the model


# Data
data = pd.DataFrame({
    'feedback': [
        'Great learning experience and friendly team',
        'The tasks were confusing and poorly explained',
        'Amazing mentorship and support',
        'No guidance at all, very frustrating',
        'I enjoyed the projects and learned a lot',
        'Very stressful environment with no help',
        'Supportive colleagues and good communication',
        'Too much pressure and unrealistic deadlines',
        'Loved the positive vibe and encouragement',
        'No proper planning or team coordination'
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
})

# Splitting the data into features and labels
#conevrting the text data into numerical data

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['feedback'])
y = data['sentiment']


# Splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Creating and training the model

model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions and checking accuracy

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Predictions:", predictions)
print("Accuracy:", accuracy)
