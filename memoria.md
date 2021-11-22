---
title: Pràctica 2 APC - GPA205-0930
author:
    - Martin Kaplan (1607076)
    - Josep Maria Domingo Catafal (1599946)
geometry: margin=3cm
numbersections: true
---

# Apartat B
![](images/B/pr-curves/LogisticRegression.png){ width=300px }
![](images/B/roc-curves/LogisticRegression.png){ width=300px }

![](images/B/pr-curves/SVC.png){ width=300px }
![](images/B/roc-curves/SVC.png){ width=300px }

![](images/B/pr-curves/KNN.png){ width=300px }
![](images/B/roc-curves/KNN.png){ width=300px }

![](images/B/pr-curves/RandomForestClassifier.png){ width=300px }
![](images/B/roc-curves/RandomForestClassifier.png){ width=300px }

# Apartat A

## Exploratory Data Analysis

Aquest dataset està destinat a crear protesis robotiques que permetin a un usuari sense braç moure la protesi al seu gust.
Per tant el gest que estigui fent aquesta persona serà el target, ja que ens interessa preveure quin gest fa la persona per així
moure el braç robotic a aquesta posició.

El dataset conte dades registrades de 8 sensors situats a l'avantbraç d'una persona mentre fa un dels 4 gestos predefinits. Aquests 4 gestos són els següents:

    0: pedra
    1: tisores
    2: paper
    3: ok

El número indica com identifiquem el gest dins del dataset.

El dataset disposa de 65 atributs 64 dels quals són les lectures dels sensors 
(disposem de 8 sensors i realitzem 8 lectures a cada sensor per tant $8 \cdot 8 = 64$).
L'altre atribut restant és el gest que està realitzant la persona. Els atributs tots són númerics.

- Podeu veure alguna correlació entre X i y?

- Estan balancejades les etiquetes (distribució similar entre categories)? Creus que pot afectar a la classificació la seva distribució?
Sí, hi ha pràcticament la mateixa quantitat de cada etiqueta:

0: 2910, 1: 2903, 2: 2943, 3: 2922


## Preprocessing

- Estàn les dades normalitzades? Caldria fer-ho? 
Cal normalitzar

- En cas que les normalitzeu, quin tipus de normalització será més adient per les vostres dades?
Estandarització


- Teniu gaires dades sense informació? Els NaNs a pandas? Tingueu en compte que hi ha metodes que no els toleren durant el aprenentatge. Com afecta a la classificació si les filtrem? I si les reompliu? Com ho farieu?
No hi ha dades sense informació ni NaNs.

- Teniu dades categoriques? Quina seria la codificació amb més sentit? (`OrdinalEncoder`, `OneHotEncoder`, d'altres?)
Totes les dades són númeriques ja que les classes ja venen codificades.

- Caldria aplicar `sklearn.decomposition.PCA`? Quins beneficis o inconvenients trobarieu?

- Es poden aplicar `PolynomialFeatures` per millorar la classificació? En quins casos té sentit fer-ho?

## Model Selection

De cara a fer les prediccions hem probat diversos models, per veure qui és el que funciona
millor amb el nostre dataset. Els models que hem probat són els següents:

- Logistic Regression
- SVC amb rbf kernel
- SVC amb linear kernel
- KNN
- Random Forest
- Perceptron
- Decision Tree

\hyperref[annex:1]{Resultats}

aaaaaaaaaaaaaaaaaaa

- Quins models heu considerat?
- Considereu les SVM amb els diferents kernels implementats? Si, rbf i linear
- Quin creieu que serà el més precís?
RBF és més precís ja que el nostre dataset no es divisible linealment.
- Quin serà el més ràpid?
- Seria una bona idea fer un ensemble? Quins inconvenients creieu que pot haver-hi?
Sería bona idea fer servir un ensamble tipus Random Forest que ja em vist que funciona
bé, però no tindria massa sentit fer un ensamble de tots els models que tenim ja que son 
models ja molt potents.

## Cross-validation

- Per què és important cross-validar els resultats?
- Separa la base de dades en el conjunt de train-test. Com de fiables serán els resultats obtinguts? En quins casos serà més fiable, si tenim moltes dades d'entrenament o poques?
- Quin tipus de K-fold heu escollit? Quants conjunts heu seleccionat (quina k)? Com afecta els diferents valors de k?
- Es viable o convenient aplicar LeaveOneOut?


## Metric Analysis 
- A teoria, hem vist el resultat d'aplicar el accuracy_score sobre dades no balancejades. Podrieu explicar i justificar quina de les següents mètriques será la més adient pel vostre problema? accuracy_score, f1_score o average_precision_score.
- Mostreu la Precisió-Recall Curve i la ROC Curve. Quina és més rellevant pel vostre dataset? Expliqueu amb les vostres paraules, la diferencia entre una i altre Pista
- Què mostra classification_report? Quina métrica us fixareu per tal de optimitzar-ne la classificació pel vostre cas?

## Hyperparameter Search

- Quines formes de buscar el millor parametre heu trobat? Són costoses computacionalment parlant?
- Si disposem de recursos limitats (per exemple, un PC durant 1 hora) quin dels dos métodes creieu que obtindrà millor resultat final?
- Existeixen altres mètodes de búsqueda més eficients (scikit-optimize)?
- Feu la prova, i amb el model i el metode de crossvalidació escollit, configureu els diferents metodes de búsqueda per a que s'executin durant el mateix temps (i.e. depenent del problema, 0,5h-1 hora). Analitzeu quin ha arribat a una millor solució. (estimeu el temps que trigarà a fer 1 training, i aixi trobeu el número de intents que podeu fer en cada cas.)

# Annex

## Resultats Model Analysis
\label{annex:1}

### Logistic Regression

                  precision    recall  f1-score   support

               0       0.53      0.47      0.50       584
               1       0.34      0.31      0.32       583
               2       0.27      0.29      0.28       596
               3       0.33      0.37      0.35       573

        accuracy                           0.36      2336
       macro avg       0.37      0.36      0.36      2336
    weighted avg       0.37      0.36      0.36      2336

### SVC rbf kernel

                  precision    recall  f1-score   support

               0       0.95      0.91      0.93       588
               1       0.88      0.98      0.93       594
               2       0.94      0.84      0.89       608
               3       0.84      0.86      0.85       546

        accuracy                           0.90      2336
       macro avg       0.90      0.90      0.90      2336
    weighted avg       0.90      0.90      0.90      2336

### SVC linear kernel

                  precision    recall  f1-score   support

               0       0.87      0.25      0.39       597
               1       0.33      0.48      0.39       597
               2       0.28      0.26      0.27       583
               3       0.30      0.40      0.34       559

        accuracy                           0.35      2336
       macro avg       0.44      0.35      0.35      2336
    weighted avg       0.45      0.35      0.35      2336

### KNN

                  precision    recall  f1-score   support

               0       0.94      0.64      0.76       575
               1       0.57      0.93      0.71       600
               2       0.78      0.32      0.46       556
               3       0.63      0.77      0.69       605

        accuracy                           0.67      2336
       macro avg       0.73      0.67      0.65      2336
    weighted avg       0.73      0.67      0.66      2336

### RandomForestClassifier

                  precision    recall  f1-score   support

               0       0.91      0.97      0.94       579
               1       0.95      0.89      0.92       602
               2       0.90      0.95      0.92       601
               3       0.88      0.84      0.86       554

        accuracy                           0.91      2336
       macro avg       0.91      0.91      0.91      2336
    weighted avg       0.91      0.91      0.91      2336

### Perceptron

                  precision    recall  f1-score   support

               0       0.29      0.39      0.34       592
               1       0.28      0.41      0.33       562
               2       0.25      0.17      0.20       604
               3       0.26      0.15      0.19       578

        accuracy                           0.28      2336
       macro avg       0.27      0.28      0.27      2336
    weighted avg       0.27      0.28      0.27      2336

### DecisionTreeClassifier

                  precision    recall  f1-score   support

               0       0.88      0.85      0.87       591
               1       0.74      0.81      0.78       553
               2       0.77      0.75      0.76       580
               3       0.71      0.70      0.70       612

        accuracy                           0.78      2336
       macro avg       0.78      0.78      0.78      2336
    weighted avg       0.78      0.78      0.78      2336