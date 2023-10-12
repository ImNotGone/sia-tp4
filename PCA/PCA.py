import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = {}

with open('../data/europe.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            data[row[0]] = [[i for i in row[1:8]]]
            line_count += 1

scaling=StandardScaler()


scaling.fit(data)
scaled_data=scaling.transform(data)

#TODO check numero de componentes
pca=PCA(n_components=7)
pca.fit(scaled_data)
x=pca.transform(scaled_data)



