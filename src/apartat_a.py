import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut

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

    # Observem les distribucions dels diferents gestos
    # Només ens fixem en les 8 primeres columnes ja que la resta són lectures
    # dels mateixos sensors una altre vegada
    sns.pairplot(dataset0.iloc[:, :8])
    plt.savefig("images/A/pairplots/gest0")

    sns.pairplot(dataset1.iloc[:, :8])
    plt.savefig("images/A/pairplots/gest1")

    sns.pairplot(dataset2.iloc[:, :8])
    plt.savefig("images/A/pairplots/gest2")

    sns.pairplot(dataset3.iloc[:, :8])
    plt.savefig("images/A/pairplots/gest3")

    # Agafem la última columna com a target i la resta com a input variables
    X = dataset.drop(dataset.columns[-1], axis=1)
    y = dataset.iloc[:, -1]

    scaler = StandardScaler()

    models = [
        LogisticRegression(random_state=0),
        svm.SVC(kernel='rbf', probability=True, random_state=0),
        svm.SVC(kernel='linear', probability=True, random_state=0),
        KNeighborsClassifier(),
        RandomForestClassifier(),
        Perceptron(),
        DecisionTreeClassifier()
    ]

    model_names = ['LogisticRegression', 'SVC', 'KNN', 'RandomForestClassifier', 'Perceptron', 'DecisionTreeClassifier']

    for model_name, model in zip(model_names, models):
        model.fit(X, y)
        print(f"{model_name} score: {model.score(X, y)}")

        # K-fold cross validation
        for k in range(2, 10):
            scores = cross_val_score(model, X, y, cv=k)
            print(f"K-fold Score for k = {k}: {scores.mean()}")

        


