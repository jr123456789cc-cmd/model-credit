# Evaluación - Modelo de Riesgo de Default

from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

DATA_DIR = Path('../data/processed')
MODEL_PATH = Path('../models/best_model.pkl')


def eval_model(file_name='credit_val.csv'):
    df = pd.read_csv(DATA_DIR / file_name).set_index('ID')
    X, y = df.drop(columns='DEFAULT'), df['DEFAULT']

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    metrics = {
        'Matriz de confusión': confusion_matrix(y, y_pred),
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
    }

    print(f'{file_name} cargado | Modelo importado correctamente')
    for name, value in metrics.items():
        print(f'{name}:\n{value}')


if __name__ == '__main__':
    eval_model()
