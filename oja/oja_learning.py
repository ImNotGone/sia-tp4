import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from activation_functions import *
from dataset_loaders import load_countries
from sklearn.metrics import mean_squared_error
import math



# Investigando encontre que Oja es solo con Identity, pero en el ppt no aclara
class SimplePerceptron:

    def __init__(self, input_size, activation_function):
        self.weights = np.random.uniform(0, 1, input_size)
        self.activation_function = activation_function

    def activation(self, inputs):
        return self.activation_function(np.dot(self.weights, inputs))


def oja_learning(data_list, column_count,initial_learning_rate, max_epoch: int = 10000):
    perceptron = SimplePerceptron(column_count, identity)
    weights_in_epochs = []
    
    for i in range(1,max_epoch):
        learning_rate = initial_learning_rate * math.exp(-0.01 * i)
        for data in data_list:
            output = perceptron.activation(data)
            delta_w = learning_rate * output * ((data) - (output  * perceptron.weights))  
            perceptron.weights += delta_w
            
        weights_in_epochs.append([np.copy(perceptron.weights), i])

    return weights_in_epochs



if __name__ == '__main__':
    input_labels, variable_labels, dataset = load_countries("./data/europe.csv")
    
    # Con learning el learning_rate tiene que ser mayor
    config_path = "./oja/config.json"
    epochs = 100
    learning_rate=0.15
    with open(config_path, "r") as f:
        config = json.load(f)
        epochs= config["epochs"]
        learning_rate=config["learning"]
    
    # Standardize data
    scaler = StandardScaler()

    scaler.fit(dataset)
    scaled_data = scaler.transform(dataset)
    weights = oja_learning(scaled_data, 7 ,learning_rate, epochs)

    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(scaled_data)  # paises en la nueva base de componentes pcpales
    print(pca.components_[0])
    print(weights[-1][0])
    
    pca_components = pca.components_[0]
    oja_components = weights[-1][0]
    
    # Por si tiene signos opuestos (sigue siendo el mismo PCA entiendo)
    if np.dot(pca_components, oja_components) < 0:
        oja_components = -oja_components

    # Calcula el error cuadrático medio (MSE) entre las componentes principales alineadas
    mse = mean_squared_error(pca_components, oja_components)

    print(f"MSE entre PCA y Oja (alineadas): {mse}")