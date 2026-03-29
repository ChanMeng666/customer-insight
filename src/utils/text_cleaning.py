import re
import jieba
import html
from typing import List, Set
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st

from src.utils import STOPWORDS_PATH

class TextCleaner:
    """Text cleaning utility class providing preprocessing, tokenization, and normalization."""

    def __init__(self, language: str = "chinese"):
        """
        Initialize the text cleaner.

        Args:
            language: Text language, supports 'chinese' or 'english'
        """
        self.language = language
        self.stopwords: Set[str] = set()
        self.load_stopwords()

    def load_stopwords(self) -> None:
        """Load the stopwords list."""
        try:
            if self.language == "chinese":
                with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                    self.stopwords = set(line.strip() for line in f)
            else:
                self.stopwords = set(stopwords.words('english'))
        except Exception as e:
            print(f"Failed to load stopwords: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text content.

        Args:
            text: Input text

        Returns:
            str: Preprocessed text
        """
        # Convert HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove emojis
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)

        return text.strip()

    def segment_text(self, text: str) -> List[str]:
        """
        Tokenize text.

        Args:
            text: Input text

        Returns:
            List[str]: List of tokens
        """
        if self.language == "chinese":
            return list(jieba.cut(text))
        else:
            return word_tokenize(text)

    def remove_stopwords(self, words: List[str]) -> List[str]:
        """
        Remove stopwords from word list.

        Args:
            words: List of tokens

        Returns:
            List[str]: Filtered word list
        """
        return [w for w in words if w not in self.stopwords]

    def clean_text(self, text: str) -> str:
        """
        Full text cleaning pipeline.

        Args:
            text: Input text

        Returns:
            str: Cleaned text
        """
        # Preprocess
        text = self.preprocess_text(text)

        # Tokenize
        words = self.segment_text(text)

        # Remove stopwords
        words = self.remove_stopwords(words)

        # Normalize
        if self.language != "chinese":
            words = [w.lower() for w in words]

        return " ".join(words)
