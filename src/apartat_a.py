import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc, average_precision_score

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
    # sns.pairplot(dataset0.iloc[:, :8])
    # plt.savefig("images/A/pairplots/gest0")

    # sns.pairplot(dataset1.iloc[:, :8])
    # plt.savefig("images/A/pairplots/gest1")

    # sns.pairplot(dataset2.iloc[:, :8])
    # plt.savefig("images/A/pairplots/gest2")

    # sns.pairplot(dataset3.iloc[:, :8])
    # plt.savefig("images/A/pairplots/gest3")

    # Agafem la última columna com a target i la resta com a input variables
    X = dataset.drop(dataset.columns[-1], axis=1)
    y = dataset.iloc[:, -1]

    # plt.figure(figsize = (10, 7))
    # sns.heatmap(dataset.iloc[:, :8].join(dataset[64]).corr(), annot=True, linewidths=.5)
    # plt.savefig("images/A/heatmap/correlationXy")

    scaler = StandardScaler()

    models = [
        LogisticRegression(max_iter=3000),
        svm.SVC(kernel='rbf', probability=True),
        svm.SVC(kernel='linear', probability=True),
        KNeighborsClassifier(),
        RandomForestClassifier(min_samples_split=20),
        Perceptron(),
        DecisionTreeClassifier()
    ]

    model_names = ['LogisticRegression', 'SVC rbf', 'SVC linear', 'KNN', 'RandomForestClassifier', 'Perceptron', 'DecisionTreeClassifier']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    n_classes = len(np.unique(y_train))

    for model_name, model in zip(model_names, models):
        if model_name == 'SVC linear': continue
        print(model_name + "\n")

        pipe = make_pipeline(scaler, model)
        pipe.fit(X_train, y_train)

        # y_test_pred = pipe.predict(X_test)
        # print(classification_report(y_test, y_test_pred))

        # # K-fold cross validation
        # for k in range(2, 10):
        #     scores = cross_val_score(model, X, y, cv=k, n_jobs=-1)
        #     print(f"K-fold Score for k = {k}: {scores.mean():.2f}")
        # print()

        if model_name != "Perceptron": # Perceptron no té predict_proba()
            probs = model.predict_proba(X_test)

            plt.figure()
            plt.title(f'Precision-Recall curve for {model_name}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_test == i, probs[:, i])
                average_precision = average_precision_score(y_test == i, probs[:, i])

                plt.plot(recall, precision,
                        label=f'Precision-recall curve of class {i} (area = {average_precision})')

            plt.legend()
            plt.savefig("images/A/pr-curves/" + model_name)

            plt.figure()
            plt.title(f'ROC curve for {model_name}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            
            # Compute ROC curve and ROC area for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test == i, probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'ROC curve of class {i} (area = {roc_auc})')

            plt.legend()
            plt.savefig("images/A/roc-curves/" + model_name)

