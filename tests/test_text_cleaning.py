import pytest
from src.utils.text_cleaning import TextCleaner


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner(language="english")

    def test_preprocess_removes_html_tags(self):
        text = "<p>Hello <b>world</b></p>"
        result = self.cleaner.preprocess_text(text)
        assert "<" not in result
        assert "Hello" in result

    def test_preprocess_removes_urls(self):
        text = "Visit https://example.com/page for info"
        result = self.cleaner.preprocess_text(text)
        assert "https://" not in result

    def test_preprocess_removes_emojis(self):
        text = "Great movie! 😀🎬"
        result = self.cleaner.preprocess_text(text)
        assert "😀" not in result

    def test_preprocess_handles_html_entities(self):
        text = "Tom &amp; Jerry"
        result = self.cleaner.preprocess_text(text)
        assert "Tom & Jerry" == result

    def test_segment_text_english(self):
        text = "Hello world test"
        result = self.cleaner.segment_text(text)
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_remove_stopwords(self):
        words = ["the", "movie", "is", "great"]
        result = self.cleaner.remove_stopwords(words)
        assert "the" not in result
        assert "movie" in result

    def test_clean_text_pipeline(self):
        text = "<p>The movie is great!</p>"
        result = self.cleaner.clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be lowercase for English
        assert result == result.lower()
