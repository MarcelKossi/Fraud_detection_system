import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.preprocessing import preprocess_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#For model training
def train_model(data_path):
    #Load and process the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data_path)
    
    #Initialization and training
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    #Predictions
    predictions = model.predict(X_test)
    
    #Evaluation of metrics
    print("\n Classification Report:\n")
    print(classification_report(y_test, predictions))   
    
   # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:\n")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    return model, scaler


# Save the trained model
def save_model(model, scaler):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nModel and scaler saved successfully.")


# Run training when script is executed
if __name__ == "__main__":

    data_path = "data/raw/creditcard.csv"

    model, scaler = train_model(data_path)

    save_model(model, scaler)