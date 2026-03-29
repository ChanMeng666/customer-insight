import jieba
import jieba.analyse
import os
import streamlit as st

from src.utils import STOPWORDS_PATH

def initialize_jieba():
    """Initialize jieba configuration"""
    stopwords_path = STOPWORDS_PATH

    # Ensure stopwords file exists
    if not os.path.exists(stopwords_path):
        raise FileNotFoundError(f"Stopwords file not found: {stopwords_path}")

    # Set jieba stopwords
    jieba.analyse.set_stop_words(stopwords_path)

    # Add custom dictionary words
    custom_words = ['唐人街', '华裔', 'PYIFF']
    for word in custom_words:
        jieba.add_word(word)

    return True
