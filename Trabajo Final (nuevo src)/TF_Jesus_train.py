# Entrenamiento - Modelo de Riesgo de Default

from pathlib import Path
import pickle
import pandas as pd
from xgboost import XGBClassifier

DATA_DIR = Path('../data/processed')
MODEL_PATH = Path('../models/best_model.pkl')


def train_model(file_name='credit_train.csv'):
    df = pd.read_csv(DATA_DIR / file_name).set_index('ID')
    X, y = df.drop(columns='DEFAULT'), df['DEFAULT']

    model = XGBClassifier(
        max_depth=2,
        n_estimators=50,
        objective='binary:logistic',
        seed=0,
        subsample=0.8,
    )
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f'{file_name} cargado | Modelo entrenado y guardado en {MODEL_PATH}')


if __name__ == '__main__':
    train_model()
