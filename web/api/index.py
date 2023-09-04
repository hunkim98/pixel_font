import json
from fastapi import FastAPI
import sys
import os
from typing import List
import numpy as np

from pydantic import BaseModel
sys.path.append('../../')


from pixel_font.common.conv_net import SimpleConvNet


app = FastAPI()

@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}

class predict_data(BaseModel):
    data: list[list[int]]
    
@app.post("/api/predict/character")
def predict_character(body: predict_data):
    cwd = os.getcwd()
    with open(f'{cwd}/api/emnist_balanced_mapping.json', 'r') as f:
        emnist_mapping = json.load(f)


    model_input = np.array(body.data).reshape(1, 1, 28, 28)
    conv_network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=len(emnist_mapping), weight_init_std=0.01)

    conv_network.load_params(f'{cwd}/api/emnist_params.pkl')

    result = conv_network.predict(model_input)
    probabilities = np.exp(result[0]) / np.sum(np.exp(result[0]))

    result_object = {}
    # probabilities = np.exp(result[0]) / np.sum(np.exp(result[0]))
    # print(probabilities)

    for key, value in emnist_mapping.items():
        character = chr(value)
        result_object[character] = probabilities[int(key)]
        # result_object[character] = result[0][key]
        # result_object[value] = result[0][key]
    
    number_to_show = 5
    # find top 3 probabilities
    top_probabilities = np.argpartition(probabilities, -number_to_show)[-number_to_show:]
    top_probabilities = top_probabilities[np.argsort(probabilities[top_probabilities])]
    # reverse the array
    top_probabilities = top_probabilities[::-1]

    for i in range(number_to_show):
        print(chr(emnist_mapping[str(top_probabilities[i])]) + ' : ' + str(probabilities[top_probabilities[i]] * 100))


    # print(chr(emnist_mapping[str(max_prob_index)]) + ' : ' + str(probabilities[max_prob_index] * 100))
    # print(result_object)

    return result_object