import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#############
# Apartat A #
#############

def apartat_a():
    # Carreguem els datasets
    dataset0 = pd.read_csv('data/0.csv', header=None)
    dataset1 = pd.read_csv('data/1.csv', header=None)
    dataset2 = pd.read_csv('data/2.csv', header=None)
    dataset3 = pd.read_csv('data/3.csv', header=None)

    # Ara els unim tots en un dataset conjunt
    dataset = pd.concat([dataset0, dataset1, dataset2, dataset3], axis=0)

    # Agafem la última columna com a target i la resta com a input variables
    X = dataset.drop(dataset.columns[-1], axis=1)
    X = StandardScaler().fit_transform(X)  # standardize the input variables
    y = dataset.iloc[:, -1]