# 🌱 Ecolight  

Ecolight is a machine learning model that predicts people’s movement across different areas of a house based on the time of day.  
It can be used to create optimized, waste-free lighting plans that reduce energy consumption.

## ✨ Motivation  

In regions where power outages are frequent, many households rely on solar energy as a backup. With limited resources, energy savings become crucial.  

Ecolight was born from this challenge:  
I wanted to help households understand and predict how they use their living spaces, so they can design smarter, cost-effective lighting strategies. By learning patterns of daily movement, Ecolight enables energy planning that is both efficient and sustainable.

## 🚀 Quick Start  

This project uses the [uv](https://docs.astral.sh/uv/) package and project manager.  

### 1. Install dependencies  
If you don’t already have **uv**, [install it](https://docs.astral.sh/uv/getting-started/installation/).  

### 2. Clone and sync  

```bash
git clone https://github.com/Onesimeav/Ecolight-Movement-AI-Model.git
cd Ecolight-Movement-AI-Model
uv sync
````

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Run the app

```bash
uv run app.py
```

Visit `http://localhost:5000` to make predictions using the default dataset.

---

## 📖 Usage

Besides the web interface, Ecolight provides scripts for training, evaluating, and analyzing models.

### Data Cleaning

Clean raw datasets before training.
This project uses data from the [CASAS Datasets](https://casas.wsu.edu/datasets/).

```bash
cd data_cleaning
uv run main.py
```

Output: `cleaned_casas_data.csv` inside `data_cleaning/data/`.

### Prepare Dataset

Place the cleaned file in a new `data/` folder as `dataset.csv`, then run:

```bash
uv run prepare_data.py
```

Outputs:

* `prepared_data.npz` (training-ready data)
* `scaler.save` (used during training & prediction)
* `location_vocab.txt`, `room_type_vocab.txt` (location & room mappings)

### Train the Model

```bash
uv run train_lstm.py
```

### Evaluate the Model

Download and clean a test dataset from CASAS. Save it as `test_dataset.csv` in `data/`, then run:

```bash
uv run model_evaluation.py
```

Outputs:

* `classification_report.csv` (evaluation report)
* `confusion_matrix.png` (visual confusion matrix)

### Analyze Movement Patterns

Generate graphs to visualize household movement patterns:

```bash
uv run analyze_patterns.py
```

Outputs graphs inside the `figure_updated/` folder.

## 🤝 Contributing

Contributions are welcome:

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

