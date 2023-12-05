import pandas as pd
import joblib


class FileManager:
    def __init__(self, filename):
        self.filename = filename

    def read_data(self):
        # Обновленный путь к файлу
        return pd.read_csv(f'Data/{self.filename}', sep=',')

    def save_results(self, data, filename):
        # Сохранение результатов в папку Data
        data.to_csv(f'Data/{filename}', index=False)

    def save_model(self, model, filename):
        # Сохранение модели в папку Data
        joblib.dump(model, f'Data/{filename}')
