# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2.0.0] - 2026-03-29

### Changed
- Translated all UI, comments, and docstrings to English
- Modernized build system from setup.py to pyproject.toml
- Reorganized project structure (utils → src/utils, data files → data/)
- Removed dead code and duplicate functions
- Updated minimum Python version to 3.9

### Added
- CI/CD pipeline with GitHub Actions (linting and testing)
- Unit test suite for core modules
- CONTRIBUTING.md guide
- GitHub issue templates (bug report, feature request)
- Centralized stopwords path configuration

## [1.0.0] - 2024-11-30

### Added
- Initial release with sentiment analysis, keyword extraction, topic modeling, and anomaly detection
- Interactive Streamlit dashboard with 6 analysis tabs
- Support for Chinese and English text analysis
- CSV and Excel file upload with data filtering
- Plotly-based interactive visualizations
