import json
from fastapi import FastAPI
import sys
sys.path.insert(0, '../..')

from common import SimpleConvNet


app = FastAPI()

@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}

@app.post("/api/predict/character")
def predict_character():
    with open('../../emnist_balanced_mapping.json', 'r') as f:
        emnist_mapping = json.load(f)
    print(emnist_mapping)

    conv_network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=47, weight_init_std=0.01)

    conv_network.load_params('emnist_params.pkl')
    return {"message": "Hello World"}