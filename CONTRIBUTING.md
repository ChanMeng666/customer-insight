# Contributing to CustomerInsight

Thank you for your interest in contributing to CustomerInsight! This guide will help you get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ChanMeng666/customer-insight.git
   cd customer-insight
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting:

```bash
ruff check .
```

### Conventions

- Use type hints for function parameters and return values
- Write docstrings in English for all public classes and methods
- Follow PEP 8 naming conventions (snake_case for functions/variables, PascalCase for classes)

## Project Structure

```
src/
├── data_processor.py    # Data loading, validation, and preprocessing
├── text_analyzer.py     # NLP analysis (sentiment, keywords, topics, insights)
├── visualizer.py        # Plotly visualization components
└── utils/
    ├── jieba_config.py  # Chinese text segmentation configuration
    └── text_cleaning.py # Text preprocessing utilities
data/
├── chinese_stopwords.txt # Chinese stopwords list
└── example_dataset.csv   # Sample dataset for testing
```

## Chinese Text Support

This project is optimized for Chinese text analysis using:
- **jieba** for Chinese word segmentation
- Custom stopwords in `data/chinese_stopwords.txt`
- BERT models fine-tuned for Chinese sentiment analysis

When modifying NLP components, ensure both Chinese and English text processing paths remain functional.

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass
4. Submit a pull request with a description of your changes

## Reporting Issues

Use [GitHub Issues](https://github.com/ChanMeng666/customer-insight/issues) to report bugs or request features.
