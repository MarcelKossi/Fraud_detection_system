# 💳 Credit Card Fraud Detection System

A production-style **Machine Learning system for detecting fraudulent credit card transactions**, combining **data science, backend APIs, and an interactive dashboard**.

This project demonstrates how a fraud detection model can be **trained, deployed, and consumed as a service**, following practices used in modern **fintech and AI-driven platforms**.

---

# 📌 Project Overview

Financial institutions process **millions of transactions every day**, and detecting fraudulent activity in real time is critical to prevent financial losses.

This project builds an **end-to-end fraud detection pipeline**:

1. Data exploration and preprocessing
2. Machine learning model training
3. Model evaluation and performance analysis
4. Deployment through a REST API
5. Interactive dashboard for transaction analysis

The system predicts whether a transaction is **legitimate or fraudulent** and provides a **fraud probability score**.

---

# 🎯 Objectives

The goal of this project is to simulate how a **modern fintech company** could build and deploy a fraud detection service.

Main objectives:

• Detect suspicious credit card transactions using machine learning
• Deploy the model as a **scalable API service**
• Provide an **interactive dashboard** for analysis and demonstration
• Follow a **modular and production-style architecture**

---

# 🧠 Machine Learning Model

The project uses a **Random Forest classifier** trained on anonymized credit card transaction data.

Key characteristics:

• Robust performance on tabular financial data
• Handles nonlinear feature relationships
• Resistant to noise and outliers
• Provides feature importance insights

### Model Performance (baseline)

| Metric    | Fraud Class |
| --------- | ----------- |
| Precision | ~0.94       |
| Recall    | ~0.82       |
| F1 Score  | ~0.87       |

Confusion Matrix Example:

```
[[56859     5]
 [   18    80]]
```

Meaning:

* **80 fraudulent transactions correctly detected**
* **18 fraud cases missed**
* **5 legitimate transactions flagged as fraud**

This represents a strong baseline model for an imbalanced fraud detection dataset.

---

# 🏗 System Architecture

The system follows a **layered ML architecture** similar to what is used in production environments.

```
User Transaction
        ↓
Streamlit Dashboard
        ↓
FastAPI REST API
        ↓
Fraud Detection Model
        ↓
Prediction + Fraud Probability
```

Each component is independent and modular.

---

# 📂 Project Structure

```
Fraud-detection-system/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   │
│   └── processed/
│
│
├── notebooks/
│   └── 01_eda.ipynb
│
│
├── src/
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── model_prediction.py
│   │
│   └── utils/
│       └── helpers.py
│
│
├── models/
│   └── fraud_model.pkl
│
│
├── api/
│   ├── main.py
│   └── schemas.py
│
│
├── dashboard/
│   └── streamlit_app.py
│
│
├── config/
│   └── config.yaml
│
│
├── tests/
│
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Technology Stack

### Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy

### Backend API

* FastAPI
* Uvicorn
* Pydantic

### Dashboard

* Streamlit

### Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Joblib

---

# 📊 Dataset

The dataset used is the **Credit Card Fraud Detection dataset**, containing anonymized transactions.

Characteristics:

* 284,807 transactions
* 492 fraud cases
* Highly imbalanced dataset (~0.17% fraud)

Features:

* V1 – V28 (anonymized PCA components)
* Time
* Amount
* Class (target variable)

```
0 → Legitimate transaction
1 → Fraudulent transaction
```

---

# 🔍 Data Processing Pipeline

The preprocessing pipeline performs the following steps:

1. Dataset loading
2. Feature / target separation
3. Stratified train-test split
4. Feature scaling using StandardScaler

Pipeline workflow:

```
Raw Dataset
      ↓
Preprocessing Pipeline
      ↓
Training / Test Sets
      ↓
Scaled Features
      ↓
Model Training
```

---

# 🚀 Running the Project

## 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/fraud-detection-system.git

cd fraud-detection-system
```

---

## 2️⃣ Create virtual environment

```
python -m venv .venv
```

Activate it:

Windows:

```
.venv\Scripts\activate
```

Mac / Linux:

```
source .venv/bin/activate
```

---

## 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 4️⃣ Train the model

```
python src/models/train_model.py
```

This will generate:

```
models/fraud_model.pkl
models/scaler.pkl
```

---

# 🌐 Running the API

Start the FastAPI server:

```
uvicorn api.main:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

Example API request:

```
POST /predict
```

Payload:

```json
{
  "features": [0.1, -0.5, 1.2, ...]
}
```

Response:

```json
{
  "prediction": 0,
  "fraud_probability": 0.02
}
```

---

# 📊 Running the Dashboard

Launch the Streamlit interface:

```
streamlit run dashboard/streamlit_app.py
```

Open in browser:

```
http://localhost:8501
```

The dashboard allows users to:

• input transaction features
• send prediction requests to the API
• visualize fraud probability

---

# 🔐 Fraud Detection Strategy

Fraud detection systems prioritize:

```
High Recall
↓
Minimize missed fraud cases
```

Trade-off considerations:

| Error Type     | Impact                              |
| -------------- | ----------------------------------- |
| False Negative | Fraud not detected (financial loss) |
| False Positive | Legitimate transaction blocked      |

Financial systems usually tolerate **more false positives** to avoid missed fraud.

---

# 🔮 Future Improvements

Possible improvements for this project:

• SMOTE or class balancing techniques
• Gradient Boosting / XGBoost models
• Hyperparameter tuning
• Feature importance visualization
• Real-time streaming pipeline
• Docker containerization
• Cloud deployment

---

# 📈 Why This Project Matters

This project demonstrates key skills required for **AI / ML engineering roles**:

* End-to-end ML pipeline development
* Backend API deployment
* Data preprocessing and model evaluation
* Interactive data product development

It simulates a simplified **fraud detection platform used in fintech environments**.

---

# 👨‍💻 Author

TAMPLARE K. Marcel
AI / Machine Learning & Backend Engineering

---

# 📜 License

This project is provided for educational and demonstration purposes.
# Fraud Detection Project

Projet de détection de fraude basé sur le dataset "creditcard" et un modèle RandomForest.

## Prérequis

- Python 3.11+

## Installation (Windows)

Depuis la racine du projet :

```bat
cd /d E:\Fraud-detection-project

python -m venv .venv
call .\.venv\Scripts\activate

python -m pip install -U pip
pip install -r requirements.txt
```

## Données

Le script d’entraînement attend le fichier :

- `data/raw/creditcard.csv`

Note : par défaut, le repo ignore les CSV dans `data/raw/` (dataset volumineux / souvent sous licence). Place le fichier localement avant d’entraîner.

## Entraînement

```bat
call .\.venv\Scripts\activate
python src\models\train_model.py
```

Cela génère (localement) :

- `models/fraud_model.pkl`
- `models/scaler.pkl`

## API (FastAPI)

Lancer l’API en mode dev :

```bat
call .\.venv\Scripts\activate
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- Swagger UI : http://127.0.0.1:8000/docs
- Health check : http://127.0.0.1:8000/
