from models.model import get_model
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train, X_test, y_test):
    model = get_model()
    print("Training model...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse:.4f}")
    return model