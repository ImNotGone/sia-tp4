import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#TODO import data from csv
data=[]

scaling=StandardScaler()


scaling.fit(data)
scaled_data=scaling.transform(data)

#TODO check numero de componentes
pca=PCA(n_components=7)
pca.fit(scaled_data)
x=pca.transform(scaled_data)



