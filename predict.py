import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('sleep_detection')

@app.route('/predict', methods=['POST'])
def predict():
    sleep_series = request.get_json()

    # TODO: create features
    X = sleep_series
    y_pred = model.predict_proba(X)[:,1]
    sleep = y_pred >= 0.5

    result = {
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
