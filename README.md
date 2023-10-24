# SIA - TP4 -  Aprendizaje No Supervisado
En este repositorio se implementaron 3 algoritmos de aprendizaje no supervisado para analizar 2 conjuntos de datos.

- Los algoritmos implementados son:
    - Kohonen
    - Oja
    - Hopfield
    - Tambien utilizamos una libreria de PCA para comparar con los resultados de Oja

## Dependencias
Para correr el proyecto es necesario tener instalado lo siguiente
    - numpy
    - scikit-learn
    - matplotlib
## Kohonen
Red de kohonen, utilizada para analisar paises europeos
### Ejecucion
Para ejecutar el programa, hay que posicionarse en el directorio `kohonen` y correr el siguiente comando

```bash
python main.py
```

El programa entrenara una red de kohonen con los parametros indicados y generara los graficos correspondientes al analisis

### Configuracion
La configuracion se encuentra en el archivo `config.json` y tiene el siguiente formato

```json
{
  "k": 3,
  "max_epochs_multiplier": 10000,
  "standardizer": "z_score",
  "initial_weights": "input",
  "radius_type": "linear",
  "radius": 3,
  "eta_type": "linear",
  "eta": 0.1,
  "distance": "euclidean",
  "run_k_test": false
}
```

- Donde se puede configurar 
    - *K*: El tamano de la red
    - *max_epochs_multiplier*: La cantidad de epocas por cada caracteristica del conjunto de datos 
    - *standardizer*: El metodo para escalar los datos, puede ser _z_score_, _min_max_ o _unit_length_
    - _initial_weights_: Si se inician los pesos de acuerdo a los inputs _inputs_ o valores aleatorios _random_
    - _radius_type_: Si se varia el radio _linear_ o es constante _constant_
    - _radius_: El radio
    - _eta_type_: Si se varia el eta _linear_ o es constante _constant_
    - _eta_: El eta
    - _distance_: Si se usa distancia euclidea _euclidean_ o  exponencial _exponential_
    - _run_k_test_: Flag que especifica si se corre un test que hace un analisis de neuronas muertas en funcion del k

## Oja
Modelo de Oja utilizada para analisis de paises europeos aproximando _PCA_

### Ejecucion
Situarse en el directorio raiz y correr

```bash
python oja/oja.py
```

o para generar los graficos de analisis 

```bash
python oja/epochs_correlation.py
python oja/epochs_learning.py
python oja/oja_learning.py
```
### Configuracion
El archivo `config.json`  tiene la siguiente forma

```json
{
    "function": "identity",
    "learning": 0.0001,
    "epochs": 1000
}
```

- Donde se puede configurar
    - _function_: Funcion de activacion del perceptron, puede ser _logistic_, _tanh_ o _identity_
    - _learning_: El learning rate
    - _epochs_: Cantidad de iteraciones

## PCA
Se usa una libreria de PCA para comparar con Oja

### Ejecucion
Situarse en el directorio `PCA` y correr

```bash
python PCA.py
```

Este programa no es ejecutable

## Hopfield
Se utiliza una red de Hopfield para analisar patrones de caracteres

### Ejecucion
Situarse en el directorio `hopfield` y correr

```bash
python main.py
```

### Configuracion
El archivo `config.json` tiene el siguiente formato
```json
{
    "limit": 100,
    "noise_percentage": 0.1,
    "display_steps": false,
    "display_energy": true,
    "selected_pattern": ["A", "G", "J", "Z"]
}
```

- Donde se puede configurar:
    - _limit_: El numero de iteraciones
    - _noise_percentage_: La tasa de ruiso
    - _display_step_: Flag para realizar un grafico de los pasos de la red
    - _display_energy_: Flag para realizar un grafico de la funcion de energia
    - _selected_pattern_: Que patrones son elegidos para guardar en la red
