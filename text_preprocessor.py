import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import cyrtranslit
class TextPreprocessor:
    def __init__(self):
        self.stemmer = SnowballStemmer('russian')
        self.stop_words = set(stopwords.words('russian'))

    def preprocess(self, text):
        text = text.lower()
        text = cyrtranslit.to_cyrillic(text, 'ru')
        text = text.replace('ё', 'е')
        text = re.sub(r'[^а-яА-ЯёЁ]', ' ', text)
        #
        words = word_tokenize(text)
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(words)

