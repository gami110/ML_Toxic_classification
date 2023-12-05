import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from file_manager import FileManager
from text_preprocessor import TextPreprocessor
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
class ToxicCommentClassifier:
    def __init__(self):
        self.file_manager = FileManager('labeled.csv')
        self.text_preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer()
        # Инициализация нескольких моделей
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'RandomForest': RandomForestClassifier(),
            'MultinomialNB': MultinomialNB(),
            #'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.pipelines = {}
        for model_name, model in self.models.items():
            self.pipelines[model_name] = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', model)
            ])

    def run(self):
        data = self.file_manager.read_data()
        X = data['comment']
        y = data['toxic'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train = X_train.apply(self.text_preprocessor.preprocess)
        X_test = X_test.apply(self.text_preprocessor.preprocess)

        param_grid = {
            'LogisticRegression': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__max_iter': [500]
            },
            'RandomForest': {
                'classifier__n_estimators': [10, 50, 100],
                'classifier__max_depth': [None, 10, 20, 30]
            },
            'MultinomialNB': {
                'classifier__alpha': [0.1, 1, 10, 100]
            }
        }
        # Обучение и оценка каждой модели, а затем сохранение
        for model_name, pipeline in self.pipelines.items():
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            print(f"Модель {model_name}:")
            print(classification_report(y_test, predictions))
            # Сохранение обученной модели
            self.file_manager.save_model(pipeline, f'{model_name}.joblib')

        for model_name, pipeline in self.pipelines.items():
            grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            print(f"Лучшие параметры для {model_name}: {grid_search.best_params_}")
            best_pipeline = grid_search.best_estimator_
            predictions = best_pipeline.predict(X_test)
            print(f"Модель {model_name} с лучшими параметрами:")
            print(classification_report(y_test, predictions))
            # Сохранение обученной модели с лучшими параметрами
            self.file_manager.save_model(best_pipeline, f'{model_name}_best.joblib')

    def load_model(self, model_name):
        # Загрузка обученного конвейера
        self.pipelines[model_name] = joblib.load(f'Data/{model_name}.joblib')
        print(f"Модель {model_name} успешно загружена.")



    def predict_comment(self, comment, model_name):
        # Предсказание с использованием загруженного конвейера
        preprocessed_comment = self.text_preprocessor.preprocess(comment)
        pipeline = self.pipelines[model_name]
        prediction = pipeline.predict([preprocessed_comment])
        return prediction
