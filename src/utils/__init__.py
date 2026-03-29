import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, 'data', 'chinese_stopwords.txt')
