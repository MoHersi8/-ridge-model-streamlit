from flask import Flask, request, jsonify
import joblib  # Ensure you have joblib to load the model

app = Flask(__name__)

# Load the trained model (make sure you have the correct path)
model = joblib.load('/Users/mohersi/Desktop/ridge_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data from POST request
    features = data['features']  # Assuming your data is in the 'features' field

    prediction = model.predict([features])  # Make prediction
    return jsonify({'prediction': prediction.tolist()})  # Return prediction as JSON

if __name__ == '__main__':
    app.run(debug=True)
