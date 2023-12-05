from sklearn.metrics import classification_report

class ModelProcessor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, text):
        preprocessed_text = self.preprocessor.preprocess(text)
        return self.model.predict([preprocessed_text])[0]

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions))