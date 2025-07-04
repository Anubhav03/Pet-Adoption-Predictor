from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and model columns
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    input_df = pd.DataFrame([data])

    # One-hot encode input data
    input_encoded = pd.get_dummies(input_df)

    # Align input data with training data columns
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale input data
    input_scaled = scaler.transform(input_aligned)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Interpret prediction
    result = "Yes! This pet is likely to be adopted!" if prediction == 1 else "This pet may not get adopted."

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
