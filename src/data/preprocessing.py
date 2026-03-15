import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load dataset
def load_dataset(path):

    df = pd.read_csv(path)
    return df


# Separate features and target
def split_features_target(df):

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


# Train-test split
def split_train_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# Feature scaling
def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# Preprocessing pipeline
def preprocess_data(data_path):

    df = load_dataset(data_path)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# Test the preprocessing script
if __name__ == "__main__":

    path = "data/raw/creditcard.csv"

    X_train, X_test, y_train, y_test, scaler = preprocess_data(path)

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)