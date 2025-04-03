from data.data_loader import load_data
from features.feature_builder import build_features
from train.train_pipeline import train_model
from utils.helpers import split_data

def main():
    df = load_data("data/sample_data.csv")
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df, target_column='target')
    train_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()