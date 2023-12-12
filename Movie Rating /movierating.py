# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the movie dataset (replace 'path/to/your/dataset.csv' with your actual dataset path)
file_path = "IMDb Movies India.csv"
movie_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(movie_data.head())

# Data preprocessing
# Assuming 'Genre', 'Director', and 'Actors' are relevant features
features = ['Genre', 'Director', 'Actors']
X = movie_data[features]
y = movie_data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature engineering and preprocessing using One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['Genre', 'Director', 'Actors'])
    ])

# Create a pipeline with the preprocessing step and a linear regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
