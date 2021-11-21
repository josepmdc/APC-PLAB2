
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

#############
# Apartat B #
#############

def apartat_b():
    X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    models = [
        LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001),
        svm.SVC(C=1.0, kernel='rbf', probability=True, random_state=0),
        KNeighborsClassifier(),
        RandomForestClassifier(),
    ]

    model_names = ['LogisticRegression', 'SVC', 'KNN', 'RandomForestClassifier']

    partitions = [0.5, 0.7, 0.8]
    for partition in partitions:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=partition, random_state=0)

        for model_name, model in zip(model_names, models):
            model.fit(X_train, y_train)
            print(f'{model_name} Score with part {partition}: {model.score(X_test, y_test)}\n')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    n_classes = len(np.unique(y_train))

    for model_name, model in zip(model_names, models):
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)

        plt.figure()
        plt.title(f'Precision-Recall curve for {model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        precision = {}
        recall = {}
        average_precision = {}

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test == i, probs[:, i])
            average_precision[i] = average_precision_score(y_test == i, probs[:, i])

            plt.plot(recall[i], precision[i],
                     label=f'Precision-recall curve of class {i} (area = {average_precision[i]})')

        plt.legend()
        plt.savefig("images/B/pr-curves/" + model_name)

        fpr = {}
        tpr = {}
        roc_auc = {}
        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        plt.title(f'ROC curve for {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # Compute and plot micro-average ROC curve and ROC area
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]})')
        plt.legend()
        plt.savefig("images/B/roc-curves/" + model_name)

    show_C_effect(X[:, :2], y, C=0.1)
    show_C_effect(X[:, :2], y, C=0.5)
    show_C_effect(X[:, :2], y, C=1.0)
    

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def show_C_effect(X, y, C=1.0, gamma=0.7, degree=3):
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    # C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C, max_iter=1000000),
              svm.SVC(kernel='rbf', gamma=gamma, C=C),
              svm.SVC(kernel='poly', degree=degree, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)

    plt.close('all')
    _, sub = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    names = datasets.load_breast_cancer().feature_names

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.savefig("images/B/C-effect/" + str(C) + ".png")
