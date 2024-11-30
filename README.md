# CustomerInsight

CustomerInsight is a comprehensive customer review analysis system that provides deep insights through sentiment analysis, keyword extraction, topic modeling, and interactive visualizations. While optimized for Chinese language reviews, it also supports English text analysis.

## Features

- 📊 Interactive data visualization and filtering
- 😊 Sentiment analysis with confidence scoring
- 🔑 Keyword extraction and trend analysis
- 📚 Topic modeling and clustering
- 🔍 Anomaly detection and insight analysis
- 📈 Custom visualization options

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages can be installed using:
```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ChanMeng666/CustomerInsight.git
cd CustomerInsight
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## Data Collection

You have two options for getting review data to analyze:

1. Use the provided example dataset:
   - Located at `CustomerInsight/example_dataset.csv`
   - Contains movie reviews with ratings, timestamps, and content
   - Perfect for testing the system's functionality

2. Generate your own dataset using [DoubanReviewScraper](https://github.com/ChanMeng666/DoubanReviewScraper):
   - A companion tool for scraping Douban movie reviews
   - Generates CSV files compatible with CustomerInsight
   - Allows you to analyze reviews for any movie on Douban

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Text Analysis**: 
  - Jieba (Chinese word segmentation)
  - Transformers (sentiment analysis)
  - scikit-learn (topic modeling)
- **Visualization**: Plotly, Matplotlib
- **Machine Learning**: scikit-learn, Isolation Forest (anomaly detection)

## Features in Detail

### Text Analysis
- Sentiment Analysis: Evaluate the emotional tone of reviews
- Keyword Extraction: Identify key terms and phrases
- Topic Modeling: Discover underlying themes in reviews
- Anomaly Detection: Flag unusual or outlier reviews

### Visualization
- Interactive time series plots
- Sentiment distribution charts
- Keyword clouds and trends
- Topic distribution visualizations
- Custom visualization options

### Data Processing
- Flexible data import (CSV, Excel)
- Advanced filtering options
- Text preprocessing and cleaning
- Statistical analysis

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors and users of the system
- Special thanks to the open source communities of Streamlit, Jieba, and other libraries used in this project

## Author

**Chan Meng**

- LinkedIn: [chanmeng666](https://www.linkedin.com/in/chanmeng666/)
- GitHub: [ChanMeng666](https://github.com/ChanMeng666)
