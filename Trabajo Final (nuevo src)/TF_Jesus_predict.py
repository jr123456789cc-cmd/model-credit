# Scoring - Modelo de Riesgo de Default

from pathlib import Path
import pickle
import pandas as pd

DATA_DIR = Path('../data/processed')
SCORES_DIR = Path('../data/scores')
MODEL_PATH = Path('../models/best_model.pkl')


def score_model(file_name='credit_score.csv', output_name='final_score.csv'):
    df = pd.read_csv(DATA_DIR / file_name).set_index('ID')

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    pred = pd.DataFrame({'PREDICT': model.predict(df)}, index=df.index)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(SCORES_DIR / output_name)

    print(f'{file_name} cargado | {output_name} exportado en {SCORES_DIR}')


if __name__ == '__main__':
    score_model()
