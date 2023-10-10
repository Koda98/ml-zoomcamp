import pickle


def load_files(model_file, dv_file):
    with open(model_file, 'rb') as f_model, open(dv_file, 'rb') as f_dv:
        model = pickle.load(f_model)
        dv = pickle.load(f_dv)
    return model, dv


def predict(client, model, dv):
    X = dv.transform(client)
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred


if __name__ == "__main__":
    model_file = "model1.bin"
    dv_file = "dv.bin"
    client = {"job": "retired", "duration": 445, "poutcome": "success"}

    model, dv= load_files(model_file, dv_file)
    credit_prob = predict(client, model, dv)
    print(credit_prob)
