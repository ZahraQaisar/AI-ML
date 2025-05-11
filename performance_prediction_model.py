from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


data = pd.DataFrame({
    'task_completion_time': [5, 8, 3],
    'feedback_rating': [4.5, 3.8, 4.9],
    'attendance': [95, 85, 98],
    'performance_score': [90, 78, 95]
})

X = data[['task_completion_time', 'feedback_rating', 'attendance']]
y = data['performance_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#  tes_size is the percentage of data to be used for testing
#  randon_state generates the same random numbers every time
print(X_train)


# Creatting and training the model
# RandomForestRegressor is a learning algorithm that uses multiple decision trees to make predictions

model = RandomForestRegressor()

#  Fit the model to the training data

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predictions: ",predictions)

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)