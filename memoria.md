---
title: Pràctica 2 APC - GPA205-0930
author:
    - Martin Kaplan (1607076)
    - Josep Maria Domingo Catafal (1599946)
geometry: margin=3cm
---

# Apartat B
![](images/pr-curves/LogisticRegression.png){ width=300px }
![](images/roc-curves/LogisticRegression.png){ width=300px }

![](images/pr-curves/SVC.png){ width=300px }
![](images/roc-curves/SVC.png){ width=300px }

![](images/pr-curves/KNN.png){ width=300px }
![](images/roc-curves/KNN.png){ width=300px }

![](images/pr-curves/RandomForestClassifier.png){ width=300px }
![](images/roc-curves/RandomForestClassifier.png){ width=300px }

# Apartat A

## Exploratory Data Analysis

Aquest dataset està destinat a crear protesis robotiques que permetin a un usuari sense braç moure la protesi al seu gust.
Per tant el gest que estigui fent aquesta persona serà el target, ja que ens interessa preveure quin gest fa la persona per així
moure el braç robotic a aquesta posició.

El dataset conte dades registrades de 8 sensors situats a l'avantbraç d'una persona mentre fa un dels 4 gestos predefinits. Aquests 4 gestos són els següents:
    - 0: pedra
    - 1: tisores
    - 2: paper
    - 3: ok

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

- Quins models heu considerat?
- Considereu les SVM amb els diferents kernels implementats.
- Quin creieu que serà el més precís?
- Quin serà el més ràpid?
- Seria una bona idea fer un ensemble? Quins inconvenients creieu que pot haver-hi?

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
