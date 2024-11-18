from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'customer_churn_prediction.pickle'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle form submission
@app.route('/', methods=['POST'])
def predict_churn():
    # Get form data
    CreditScore = int(request.form['CreditScore'])
    Geography = request.form['Geography']
    Gender = request.form['Gender']
    Age = int(request.form['Age'])
    Balance = float(request.form['Balance'])
    NumOfProducts = int(request.form['NumOfProducts'])
    IsActiveMember = int(request.form['IsActiveMember'])

    # Preprocess input data
    Geography = 0 if Geography == "France" else 1 if Geography == "Germany" else 2
    Gender = 1 if Gender == "Male" else 0  # Encode Male as 1, Female as 0

    # Create a numpy array for the model input
    features = np.array([[CreditScore, Geography, Gender, Age, Balance, NumOfProducts, IsActiveMember]])

    # Predict based on features
    prediction = model.predict(features)
    churn_prediction = "Yes" if prediction[0] == 1 else "No"

    # Render the result with a clear message for the prediction
    message = f"The model predicts that the customer will {'churn' if churn_prediction == 'Yes' else 'not churn'}."

    return render_template('index.html', churn_prediction=message)

if __name__ == "__main__":
    app.run(debug=True)