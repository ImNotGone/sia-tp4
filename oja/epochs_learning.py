import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from activation_functions import *
from dataset_loaders import load_countries
from oja import oja
from oja_learning import oja_learning

def run_oja_epochs(data, num_epochs, initial_learning_rate, repetitions, epoch_interval):
    correlations = []

    pca = PCA(n_components=1)
    pca.fit(data)
    pca_components = pca.components_[0]

    for i in range(2, num_epochs + 1):
        if i % epoch_interval == 0:
            correlations_epoch = []

            for _ in range(repetitions):
                weights = oja_learning(data, 7, initial_learning_rate, i)
                oja_components = weights[-1][0]

                # Ajusta el signo si el producto escalar es negativo
                if np.dot(pca_components, oja_components) < 0:
                    oja_components = -oja_components

                # Calcula la correlación entre los componentes principales
                correlation = np.corrcoef(pca_components, oja_components)[0, 1]
                correlations_epoch.append(correlation)
                
            # Calcula el promedio de las correlaciones para esta época
            correlation_avg = np.mean(correlations_epoch)
            correlations.append(correlation_avg)
        else:
            # Agrega None para las épocas intermedias sin datos
            correlations.append(None)

    return correlations, oja_components, pca_components

if __name__ == '__main__':
    input_labels, variable_labels, dataset = load_countries("./data/europe.csv")
    scaler = StandardScaler()
    scaler.fit(dataset)
    scaled_data = scaler.transform(dataset)
    num_epochs = 100  # Número total de épocas
    initial_learning_rate = 0.12  # Tasa de aprendizaje inicial
    repetitions = 5  # Número de repeticiones en cada época
    epoch_interval = 5  # Intervalo de épocas para procesar los datos
    
    # Ejecuta Oja a lo largo de las épocas y guarda las correlaciones
    correlations_avg, ojac, pcac = run_oja_epochs(scaled_data, num_epochs, initial_learning_rate, repetitions, epoch_interval)

    
    x = [i for i in range(2, num_epochs + 1)]
    plt.plot(x, correlations_avg, marker='o', linestyle='-', markersize=3)
    plt.xlabel("Época")
    plt.ylabel("Promedio de Correlación")
    plt.title("Promedio de Correlación entre componentes principales de PCA y Oja")
    for i, (pca_value, oja_value) in enumerate(zip(pcac, ojac)):
        plt.text(num_epochs - 5000, 0.6 - i * 0.03, f'PCA[{i}]: {pca_value:.4f}', fontsize=8, color='red')
        plt.text(num_epochs - 2000, 0.6 - i * 0.03, f'Oja[{i}]: {oja_value:.4f}', fontsize=8, color='blue')
    plt.show()
