from flask import Flask, request, jsonify
from sklearn.utils import parallel_backend
import joblib
import numpy as np

app = Flask('startup_prediction')


with parallel_backend('multiprocessing', n_jobs=-1):
    model = joblib.load('regression_model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data as a dictionary
    request_data = request.get_json()

    # Extract the parameters from the request data
    milestones = request_data['milestones']
    is_first_round = request_data['is_first_round']
    post_money_valuation_usd = request_data['post_money_valuation_usd']
    degree_type = request_data['degree_type']
    country_code = request_data['country_code']

    # Convert the one-hot encoded variables to a numpy array
    degree_type_array = np.array(degree_type).reshape(1, -1)
    country_code_array = np.array(country_code).reshape(1, -1)

    # Concatenate the parameters into a single numpy array for prediction
    X = np.array([milestones, is_first_round, post_money_valuation_usd])
    X = np.concatenate((X.reshape(1, -1), degree_type_array,
                       country_code_array), axis=1)

    # Make the prediction using the loaded model
    prediction = model.predict(X)

    # Return the prediction as a JSON response
    return {'prediction': prediction.tolist()}


# milestones(INT),is_first_round(BOOL),post_money_valuation_usd(float),degree_type(one hot encoding), country_code(one hot encoding)
if __name__ == '__main__':
    app.run(port=5000, debug=True)
