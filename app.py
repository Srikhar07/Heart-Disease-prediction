from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('heart_disease_prediction.joblib')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Step 1: Get all input values from the form
    input_values = request.form.values()  # This gets all inputs as strings

    # Step 2: Convert input strings to floats
    input_floats = []
    for value in input_values:
        input_floats.append(float(value))

    # Step 3: Convert to a NumPy array (required by the model)
    features = np.array([input_floats])

    # Step 4: Make prediction using the loaded model
    prediction = model.predict(features)

    # Step 5: Decide the output based on prediction
    if prediction[0] == 1:
        output = 'Heart Disease Detected 💔'
    else:
        output = 'No Heart Disease ❤️'

    # Step 6: Render the HTML page with the prediction result
    return render_template('home.html', prediction_text='Result: ' + output)


if __name__ == "__main__":
    app.run(debug=True)
