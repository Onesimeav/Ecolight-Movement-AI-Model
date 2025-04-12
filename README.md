# Ecolight-Movement-AI-Model

## Lancer l'application

Pour exécuter l'application, vous devez lancer le fichier `app.py` :

```bash
python app.py
```

Ce script lancera un serveur local accessible à l'adresse suivante :  
**[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

Une fois sur cette URL, vous pouvez utiliser l’interface web pour **générer des prédictions** via l’application.

---

## Évaluer le modèle

Pour tester et évaluer le modèle existant, vous pouvez utiliser directement le script suivant :

```bash
python model_evaluation.py
```

Ce script va effectuer une évaluation du modèle actuel.

---

## Reprendre tout le processus depuis le début

Si vous souhaitez tout reprendre depuis le début, suivez ces étapes :

### 1. Nettoyage des données

Rendez-vous dans le dossier `Data_Cleaning` et lancez le script `main.py` :

```bash
cd Data_Cleaning
python main.py
```

Par défaut, ce script utilise un fichier nommé `dirty_data.csv`.  
Si vous le souhaitez, vous pouvez **remplacer ce fichier** par vos propres données en le téléchargeant depuis le site de **CASAS** :  
➡️ [https://data.casas.wsu.edu/download/](https://data.casas.wsu.edu/download/)

Ce script va générer un nouveau fichier appelé :

```
Cleaned_Casas_data.csv
```

### 2. Préparation des données

Créez un dossier `Data` à la racine du projet et **déplacez** le fichier `Cleaned_Casas_data.csv` dedans.  
Renommez ensuite ce fichier en :

```
Datasets.csv
```

Lancez maintenant le script `prepare_data_from_txt.py` :

```bash
python prepare_data_from_txt.py
```

Ce script va générer un fichier `.npz` à partir des données nettoyées.

### 3. Entraînement du modèle

Une fois les données prêtes, vous pouvez entraîner le modèle en lançant :

```bash
python trainlstm.py
```



