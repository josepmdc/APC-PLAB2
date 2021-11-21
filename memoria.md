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

- **Quants atributs té la vostra base de dades?**

64

- **Quin tipus d'atributs tens? (Númerics, temporals, categorics, binaris...)**

Númerics. Cada atribut representa el valor llegit per un sensor.

- **Com es el target, quantes categories diferents existeixen?**

El target és el gest de la mà. Existeixen 4 valors numerics diferents i cada un representa un gest:
    - 0: pedra
    - 1: tisores
    - 2: paper
    - 3: ok

- **Podeu veure alguna correlació entre X i y?**


- **Estan balancejades les etiquetes (distribució similar entre categories)? Creus que pot afectar a la classificació la seva distribució?**

Sí, hi ha pràcticament la mateixa quantitat de cada etiqueta:

0: 2910, 1: 2903, 2: 2943, 3: 2922