import pickle
from flask import Flask, request, jsonify


app = Flask('credit')


def load_files(model_file, dv_file):
    with open(model_file, 'rb') as f_model, open(dv_file, 'rb') as f_dv:
        model = pickle.load(f_model)
        dv = pickle.load(f_dv)
    return model, dv


@app.route('/predict', methods=['POST'])
def predict():
    model_file = "model2.bin"
    dv_file = "dv.bin"
    client = request.get_json()

    model, dv= load_files(model_file, dv_file)
    X = dv.transform(client)
    credit_prob = model.predict_proba(X)[0, 1]

    result = {'credit_probability': float(credit_prob)}
    return jsonify(result)


@app.route('/', methods=['GET'])
def index():
    return "Hello World"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
