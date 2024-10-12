from flask import Flask, request, jsonify
from joblib import load
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

model = load('loyalty_predict_simple_reg.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if 'input_data' not in data:
            return jsonify({"error": "Missing 'input_data' key in request body"}), 400

        input_data = np.array(data['input_data'])

        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)

        prediction = model.predict(input_data)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
