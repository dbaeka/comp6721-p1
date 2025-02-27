import os

import joblib


def save_feature_transform(transform, name):
    os.makedirs('feature_transforms', exist_ok=True)

    snake_name = name.replace(" ", "_").lower()
    joblib.dump(transform, f'feature_transforms/{snake_name}.joblib')

    print("Transform successfully saved to the 'feature_transforms' directory")


def save_models(model, name):
    os.makedirs('models', exist_ok=True)

    snake_name = name.replace(" ", "_").lower()
    joblib.dump(model, f'models/{snake_name}.joblib')

    print("Models successfully saved to the 'models' directory")


def load_model(model_name):
    snake_name = model_name.replace(" ", "_").lower()
    model = joblib.load(f'models/{snake_name}.joblib')

    return model


def load_feature_transform(transform_name):
    snake_name = transform_name.replace(" ", "_").lower()
    transform = joblib.load(f'feature_transforms/{snake_name}.joblib')

    return transform
