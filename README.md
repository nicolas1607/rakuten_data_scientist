# Rakuten

### Pré-requis
Python 3.8.10

## Installer les librairies
```
> cd mai24_bds_rakuten
> python3 -m venv env
> source env/bin/activate
> pip install -r requirements.txt
```

## Fichiers .zip fournis
Ces fichiers sont fournis en archive afin d'éviter de relancer le pre-processing des données et les modèles LinearSVM et ResNet50.

> unzip les données

**Données fournies par Rakuten**
>data/X_train.csv\
data/y_train.csv\
data/images/image_train\
data/images/image_test

**Données fournies pour le pre-processing**
>data/df_lang_preprocessed.csv\
data/df_cleaned.csv\
data/df_lemmatized.csv\
data/df_tokenized.csv\
data/X_train_sampled.npz\
data/y_train_sampled.csv\
data/images/image_train_preprocessed

**Données fournies pour les modèles retenus**
>models/linear_svm.pkl \
models/resnet50.pkl
models/resnet50_history.pkl


## Rapport projet
Le rapport final du projet en PDF
>reports/rapport_projet_rakuten.pdf

## Graphiques
Les matrices de confusion
>reports/figures/matrice_de_confusion

Les nuages de mots
>reports/figures/nuage_de_mot

## Exploration des données & Data Vizualisation
```
> cd mai24_bds_rakuten
> python3 src/visualization/__init__.py
```


## Pre-processing & Modélisation
```
> cd mai24_bds_rakuten
> python3 src/models/__init__.py
```

==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
