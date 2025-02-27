import os

import joblib


def save_feature_scaler(scaler):
    os.makedirs('models', exist_ok=True)

    joblib.dump(scaler, 'models/feature_scaler.joblib')
    print("Feature scaler successfully saved to the 'models' directory")


def save_models(model, name):
    os.makedirs('models', exist_ok=True)

    snake_name = name.replace(" ", "_").lower()
    joblib.dump(model, f'models/{snake_name}.joblib')

    print("Models successfully saved to the 'models' directory")


def load_model(model_name):
    snake_name = model_name.replace(" ", "_").lower()
    model = joblib.load(f'models/{snake_name}.joblib')

    return model


def load_feature_scaler():
    return joblib.load('models/feature_scaler.joblib')
