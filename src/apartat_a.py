import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut

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
        LogisticRegression(max_iter=3000),
        svm.SVC(kernel='rbf'),
        svm.SVC(kernel='linear'),
        KNeighborsClassifier(),
        RandomForestClassifier(min_samples_split=20),
        Perceptron(),
        DecisionTreeClassifier()
    ]

    model_names = ['LogisticRegression', 'SVC rbf', 'SVC linear', 'KNN', 'RandomForestClassifier', 'Perceptron', 'DecisionTreeClassifier']

    for model_name, model in zip(model_names, models):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        pipe = make_pipeline(scaler, model)
        pipe.fit(X_train, y_train)

        print(model_name + "\n")

        y_test_pred = pipe.predict(X_test)
        print(classification_report(y_test, y_test_pred))

        # K-fold cross validation
        # for k in range(2, 10):
        #     scores = cross_val_score(model, X, y, cv=k)
        #     print(f"K-fold Score for k = {k}: {scores.mean():.2f}")
        # print()
        # # Leave-one-out cross validation
        # loo = LeaveOneOut()
        # scores = cross_val_score(model, X, y, cv=loo)
        # print(f"Leave-one-out Score: {scores.mean()}")
