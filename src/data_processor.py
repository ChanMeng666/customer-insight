import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import jieba
import jieba.analyse
import os
import streamlit as st

from src.utils import STOPWORDS_PATH

class DataProcessor:
    """Data processing class for loading, validating, cleaning, and analyzing data."""

    REQUIRED_COLUMNS = ['content', 'rating', 'timestamp']

    def __init__(self):
        """Initialize the data processor."""
        self.data: Optional[pd.DataFrame] = None
        self.stats: Dict = {}

        # Initialize jieba
        self._initialize_jieba()

    def _initialize_jieba(self):
        """Initialize jieba tokenizer and custom dictionary."""
        try:
            jieba.initialize()

            # Set stopwords file path
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            stopwords_path = STOPWORDS_PATH

            # Ensure stopwords file exists
            if os.path.exists(stopwords_path):
                jieba.analyse.set_stop_words(stopwords_path)
            else:
                st.warning(f"Stopwords file not found: {stopwords_path}")

            # Add custom dictionary words
            custom_words = [
                '唐人街', '华裔', 'PYIFF', '映后', '纪录片',
                '平遥', '导演', '影评', '观众', '剧情'
            ]
            for word in custom_words:
                jieba.add_word(word)

        except Exception as e:
            st.error(f"Failed to initialize jieba: {str(e)}")

    def load_data(self, file_path: str) -> bool:
        """
        Load a data file and perform initial validation.

        Args:
            file_path: Path to the data file

        Returns:
            bool: Whether the data was loaded successfully
        """
        try:
            dtype_dict = {
                'content': str,
                'rating': float,
                'user_id': str,
                'category': str
            }

            if file_path.endswith('.csv'):
                self.data = pd.read_csv(
                    file_path,
                    dtype=dtype_dict
                )
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(
                    file_path,
                    dtype=dtype_dict
                )
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files")

            # Validate required columns
            if not self._validate_columns():
                raise ValueError(f"Data is missing required columns: {', '.join(self.REQUIRED_COLUMNS)}")

            # Process timestamps
            self._process_timestamp()

            # Clean data
            self.clean_data()

            return True

        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return False

    def _process_timestamp(self):
        """Process the timestamp column."""
        try:
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(
                    self.data['timestamp'],
                    format='%Y/%m/%d %H:%M',
                    errors='coerce'
                )

                # Check for invalid timestamps
                invalid_dates = self.data['timestamp'].isnull().sum()
                if invalid_dates > 0:
                    st.warning(f"Found {invalid_dates} records with invalid timestamps")

                # Drop invalid timestamps
                self.data = self.data.dropna(subset=['timestamp'])

        except Exception as e:
            st.error(f"Failed to process timestamps: {str(e)}")

    def _validate_columns(self) -> bool:
        """
        Validate that the data contains all required columns.

        Returns:
            bool: Whether validation passed
        """
        return all(col in self.data.columns for col in self.REQUIRED_COLUMNS)

    def clean_data(self) -> bool:
        """
        Clean the data by handling null values and outliers.

        Returns:
            bool: Whether cleaning was successful
        """
        if self.data is None:
            return False

        try:
            # Drop null values
            original_length = len(self.data)
            self.data = self.data.dropna(subset=['content', 'rating'])
            dropped_rows = original_length - len(self.data)
            if dropped_rows > 0:
                st.warning(f"Dropped {dropped_rows} records with null values")

            # Handle rating outliers
            self.data = self.data[self.data['rating'].between(1, 5)]

            # Clean text content
            self.data['content'] = self.data['content'].astype(str).apply(self._clean_text)

            # Add text length column
            self.data['text_length'] = self.data['content'].str.len()

            # Add default category if missing
            if 'category' not in self.data.columns:
                self.data['category'] = 'default'

            # Reset index
            self.data = self.data.reset_index(drop=True)

            return True

        except Exception as e:
            st.error(f"Data cleaning error: {str(e)}")
            return False

    def _clean_text(self, text: str) -> str:
        """
        Clean text content.

        Args:
            text: Input text

        Returns:
            str: Cleaned text
        """
        import re

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters (keep Chinese characters)
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)

        return text.strip()

    def calculate_statistics(self) -> Dict:
        """
        Calculate basic statistics.

        Returns:
            Dict: Dictionary containing various statistical metrics
        """
        if self.data is None:
            return {}

        try:
            self.stats = {
                'total_reviews': len(self.data),
                'average_rating': round(self.data['rating'].mean(), 2),
                'rating_distribution': self.data['rating'].value_counts().to_dict(),
                'daily_reviews': self.data.groupby(
                    self.data['timestamp'].dt.date
                ).size().to_dict(),
                'category_distribution': self.data['category'].value_counts().to_dict(),
                'text_length_stats': {
                    'mean': int(self.data['text_length'].mean()),
                    'median': int(self.data['text_length'].median()),
                    'min': int(self.data['text_length'].min()),
                    'max': int(self.data['text_length'].max())
                }
            }

            return self.stats

        except Exception as e:
            st.error(f"Statistics calculation error: {str(e)}")
            return {}

    def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Filter data by date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            pd.DataFrame: Filtered data
        """
        if self.data is None:
            return pd.DataFrame()

        try:
            mask = (self.data['timestamp'].dt.date >= start_date.date()) & \
                   (self.data['timestamp'].dt.date <= end_date.date())
            return self.data[mask]

        except Exception as e:
            st.error(f"Date filtering error: {str(e)}")
            return pd.DataFrame()

    def filter_by_rating(self, min_rating: float, max_rating: float) -> pd.DataFrame:
        """
        Filter data by rating range.

        Args:
            min_rating: Minimum rating
            max_rating: Maximum rating

        Returns:
            pd.DataFrame: Filtered data
        """
        if self.data is None:
            return pd.DataFrame()

        try:
            return self.data[
                (self.data['rating'] >= min_rating) &
                (self.data['rating'] <= max_rating)
            ]
        except Exception as e:
            st.error(f"Rating filtering error: {str(e)}")
            return pd.DataFrame()
