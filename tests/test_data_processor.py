import pytest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor


class TestDataProcessor:
    def setup_method(self):
        self.processor = DataProcessor()

    def test_validate_columns_valid(self, sample_df):
        self.processor.data = sample_df
        assert self.processor._validate_columns() is True

    def test_validate_columns_missing(self):
        self.processor.data = pd.DataFrame({'foo': [1], 'bar': [2]})
        assert self.processor._validate_columns() is False

    def test_clean_text_removes_html(self):
        text = "<p>Hello <b>world</b></p>"
        result = self.processor._clean_text(text)
        assert "<" not in result
        assert ">" not in result

    def test_clean_text_removes_urls(self):
        text = "Check out https://example.com for more"
        result = self.processor._clean_text(text)
        assert "https://" not in result

    def test_clean_text_removes_extra_whitespace(self):
        text = "Hello    world   test"
        result = self.processor._clean_text(text)
        assert "  " not in result

    def test_clean_data_drops_nulls(self, sample_df):
        df = sample_df.copy()
        df.loc[0, 'content'] = None
        self.processor.data = df
        self.processor.clean_data()
        assert len(self.processor.data) == len(sample_df) - 1

    def test_clean_data_filters_ratings(self, sample_df):
        df = sample_df.copy()
        df.loc[0, 'rating'] = 6.0  # Out of range
        self.processor.data = df
        self.processor.clean_data()
        assert self.processor.data['rating'].max() <= 5.0

    def test_clean_data_adds_text_length(self, sample_df):
        self.processor.data = sample_df.copy()
        self.processor.clean_data()
        assert 'text_length' in self.processor.data.columns

    def test_calculate_statistics(self, sample_df):
        self.processor.data = sample_df.copy()
        self.processor.data['text_length'] = self.processor.data['content'].str.len()
        stats = self.processor.calculate_statistics()
        assert stats['total_reviews'] == 5
        assert 'average_rating' in stats
        assert 'rating_distribution' in stats
        assert 'daily_reviews' in stats

    def test_calculate_statistics_empty(self):
        self.processor.data = None
        assert self.processor.calculate_statistics() == {}

    def test_filter_by_rating(self, sample_df):
        self.processor.data = sample_df.copy()
        result = self.processor.filter_by_rating(3.0, 5.0)
        assert all(result['rating'] >= 3.0)
        assert all(result['rating'] <= 5.0)

    def test_filter_by_rating_empty(self):
        self.processor.data = None
        result = self.processor.filter_by_rating(1.0, 5.0)
        assert result.empty
