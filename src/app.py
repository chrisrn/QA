from flask import Flask,request, jsonify, json
from flask_cors import CORS

from transformers import DistilBertConfig
from inference import SquadModel

app = Flask(__name__)
CORS(app)

experiment_path = 'results/2023_08_17_15_04_00/exp_0'
with open(f'{experiment_path}/config.json') as json_file:
    exp = json.load(json_file)['hyper_parameters']
config = DistilBertConfig(max_position_embeddings=512,
                          n_layers=exp['num_hidden_layers'],
                          n_heads=exp['num_heads'],
                          dim=exp['dim_encoder'],
                          hidden_dim=exp['hidden_trans_dim'],
                          dropout=exp['dropout'],
                          activation=exp['activation'])
model = SquadModel(config)
model.load_model(f'{experiment_path}/model/best_model.pth.tar')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    cont = request.json["context"]
    q = request.json["question"]
    try:
        out = model.predict(cont, q)
        return jsonify({"result": out})
    except Exception as e:
        print(e)
        return jsonify({"result": "Model Failed"})


if __name__ == "__main__":
    app.run('0.0.0.0', port=8000)
