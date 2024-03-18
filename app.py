from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the data
data = pd.read_csv("movies_youtube_sentiments.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
data['genre_encoded'] = label_encoder.fit_transform(data['genre'])
data['country_encoded'] = label_encoder.fit_transform(data['country'])

# Separate numeric and categorical columns
numeric_columns = ['year', 'votes', 'budget', 'runtime']
categorical_columns = ['genre', 'country']

# Handle missing values for numeric columns
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

X = data[['year', 'votes', 'budget', 'runtime', 'genre_encoded', 'country_encoded']]
y = data['favorability']

# Create models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

# Fit the models
for model_name, model in models.items():
    model.fit(X, y)

# Define a default prediction value or strategy to handle unseen labels
default_prediction = 0  # Example: Default prediction is set to 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            votes = int(request.form['votes'])
            budget = int(request.form['budget'])
            runtime = int(request.form['runtime'])
            genre = request.form['genre']
            country = request.form['country']

            # Encode categorical variables using the same label encoder instance
            genre_encoded = encode_categorical(label_encoder, genre)
            country_encoded = encode_categorical(label_encoder, country)

            # Make predictions
            predictions = {}
            for model_name, model in models.items():
                prediction = make_prediction(model, year, votes, budget, runtime, genre_encoded, country_encoded)
                predictions[model_name] = prediction

            return render_template('results.html', predictions=predictions)
        except Exception as e:
            return render_template('error.html', error=str(e))

def encode_categorical(label_encoder, value):
    try:
        encoded_value = label_encoder.transform([value])[0]
    except ValueError:
        # If value is unseen during prediction, return default value or handle accordingly
        print(f"Unseen label during prediction: {value}")
        encoded_value = default_prediction
    return encoded_value

def make_prediction(model, year, votes, budget, runtime, genre_encoded, country_encoded):
    # Make prediction using the given model
    prediction = model.predict([[year, votes, budget, runtime, genre_encoded, country_encoded]])
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)
