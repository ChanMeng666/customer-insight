from typing import List, Dict, Optional
import re
import torch
from torch.nn.functional import softmax
import streamlit as st
from src.utils import STOPWORDS_PATH
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import jieba.analyse
import numpy as np
from datetime import datetime
from typing import Set, Dict, List
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime, timedelta
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')  # Download necessary resources for word_tokenize

@st.cache_data
def load_stopwords(language: str) -> Set[str]:
    """
    Load stopwords with caching.

    Args:
        language: Text language

    Returns:
        Set[str]: Set of stopwords
    """
    try:
        if language == 'chinese':
            with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        else:
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
    except Exception as e:
        st.warning(f"Failed to load stopwords: {str(e)}")
        return set()

class TextAnalyzer:
    """Base class for text analysis."""

    def __init__(self, language: str = 'chinese'):
        """
        Initialize the text analyzer.

        Args:
            language: Text language, supports 'chinese' or 'english'
        """
        self.language = language

class SentimentAnalyzer(TextAnalyzer):
    """Text sentiment analyzer."""

    def __init__(self, language: str = 'chinese'):
        """
        Initialize the sentiment analyzer.

        Args:
            language: Text language, supports 'chinese' or 'english'
        """
        super().__init__(language)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Select model
        self.model_name = 'uer/roberta-base-finetuned-jd-binary-chinese' if language == 'chinese' \
                         else 'nlptown/bert-base-multilingual-uncased-sentiment'

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            raise
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Analyze sentiment of texts in batches.

        Args:
            texts: List of texts
            batch_size: Batch size

        Returns:
            List[Dict]: List of sentiment analysis results
        """
        results = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Predict
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = softmax(outputs.logits, dim=-1)

                # Process results
                for j, (text, pred) in enumerate(zip(batch_texts, predictions)):
                    if self.language == 'chinese':
                        # Chinese model: binary classification (positive/negative)
                        sentiment = 'positive' if pred[1] > pred[0] else 'negative'
                        confidence = float(pred.max())
                    else:
                        # English model: 5-class classification
                        sentiment_score = int(torch.argmax(pred)) + 1
                        sentiment = self._get_sentiment_label(sentiment_score)
                        confidence = float(pred.max())

                    results.append({
                        'text': text,
                        'sentiment': sentiment,
                        'confidence': confidence
                    })

                # Update progress
                if hasattr(st, 'progress_bar'):
                    progress = (i + len(batch_texts)) / len(texts)
                    st.progress_bar.progress(progress)

        except Exception as e:
            st.error(f"Error during sentiment analysis: {str(e)}")
            raise

        return results
    
    def _get_sentiment_label(self, score: int) -> str:
        """
        Convert a score to a sentiment label.

        Args:
            score: Score from 1 to 5

        Returns:
            str: Sentiment label
        """
        if score <= 2:
            return 'negative'
        elif score == 3:
            return 'neutral'
        else:
            return 'positive'
    
    def get_sentiment_stats(self, results: List[Dict]) -> Dict:
        """
        Calculate sentiment analysis statistics.

        Args:
            results: List of sentiment analysis results

        Returns:
            Dict: Statistics
        """
        df = pd.DataFrame(results)

        stats = {
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'average_confidence': float(df['confidence'].mean()),
            'typical_positive': df[df['sentiment'] == 'positive'].nlargest(3, 'confidence')[['text', 'confidence']].to_dict('records'),
            'typical_negative': df[df['sentiment'] == 'negative'].nlargest(3, 'confidence')[['text', 'confidence']].to_dict('records')
        }

        return stats
    
    @staticmethod
    @st.cache_data
    def cached_analyze_batch(texts: List[str], model_name: str, device: str, language: str, batch_size: int = 32) -> List[Dict]:
        """
        Cached batch text sentiment analysis.

        Args:
            texts: List of texts
            model_name: Model name
            device: Device type
            language: Text language
            batch_size: Batch size

        Returns:
            List[Dict]: List of sentiment analysis results
        """
        analyzer = SentimentAnalyzer(language)
        return analyzer.analyze_batch(texts, batch_size)


class KeywordAnalyzer(TextAnalyzer):
    def __init__(self, language: str = 'chinese'):
        super().__init__(language)
        self.stop_words = load_stopwords(language)

        # Add review-related stopwords
        self.stop_words.update({'电影', '片子', '导演', '拍摄', '#', 'PYIFF'})

        # Configure jieba tokenizer
        import jieba.analyse
        jieba.analyse.set_stop_words(STOPWORDS_PATH)

        # Custom dictionary
        for word in ['唐人街', '华裔']:
            jieba.add_word(word)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text."""
        # Remove hashtag labels
        text = re.sub(r'#.*?#', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        return text.strip()

    def extract_keywords(self, texts: List[str], top_n: int = 20) -> Dict[str, float]:
        """Extract keywords and their weights."""
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            combined_text = ' '.join(processed_texts)

            # Extract keywords using jieba
            keywords = {}
            for kw, weight in jieba.analyse.extract_tags(
                combined_text,
                topK=top_n,
                withWeight=True,
                allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn')  # Allow more POS tags
            ):
                if kw not in self.stop_words and len(kw) > 1:  # Filter single-character words
                    keywords[kw] = float(weight)

            # Normalize weights
            if keywords:
                max_weight = max(keywords.values())
                keywords = {k: float(v/max_weight) for k, v in keywords.items()}

            return keywords

        except Exception as e:
            st.error(f"Failed to extract keywords: {str(e)}")
            return {}
    
    def calculate_keyword_trends(self, df: pd.DataFrame,
                               top_keywords: List[str],
                               time_window: str = 'M') -> pd.DataFrame:
        """Calculate keyword trends over time."""
        try:
            df = df.copy()

            # Ensure timestamp column is datetime type
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M')

            # Group by time window and category
            grouped = df.groupby([
                pd.Grouper(key='timestamp', freq=time_window),
                'category'
            ])

            trend_data = []

            for (time, category), group in grouped:
                if len(group) == 0:
                    continue

                # Extract keywords
                texts = group['content'].tolist()
                keywords = self.extract_keywords(texts, len(top_keywords))

                # Record frequencies
                for keyword in top_keywords:
                    trend_data.append({
                        'timestamp': time,
                        'category': category,
                        'keyword': keyword,
                        'frequency': keywords.get(keyword, 0.0)
                    })

            return pd.DataFrame(trend_data)

        except Exception as e:
            st.error(f"Failed to analyze trends: {str(e)}")
            return pd.DataFrame()
    
    def extract_keywords_by_rating(self, df: pd.DataFrame, top_n: int = 20) -> Dict[str, Dict[str, float]]:
        """Extract keywords grouped by rating."""
        try:
            df = df.copy()

            # Group by rating and sentiment category
            positive_mask = (df['rating'] >= 4) | (df['category'].str.contains('positive', case=False))
            negative_mask = (df['rating'] <= 2) | (df['category'].str.contains('negative', case=False))

            positive_texts = df[positive_mask]['content'].tolist()
            negative_texts = df[negative_mask]['content'].tolist()

            # Extract keywords
            positive_keywords = self.extract_keywords(positive_texts, top_n) if positive_texts else {}
            negative_keywords = self.extract_keywords(negative_texts, top_n) if negative_texts else {}

            return {
                'positive': positive_keywords,
                'negative': negative_keywords
            }

        except Exception as e:
            st.error(f"Failed to analyze keywords by rating: {str(e)}")
            return {'positive': {}, 'negative': {}}

class TopicAnalyzer(TextAnalyzer):
    """Topic analyzer."""

    def __init__(self, language: str = 'chinese'):
        """
        Initialize the topic analyzer.

        Args:
            language: Text language, supports 'chinese' or 'english'
        """
        super().__init__(language)

        # Initialize models
        self.lda_model = None
        self.kmeans_model = None

        # Load stopwords
        self.stop_words = self.load_stopwords()

        # Initialize vectorizer
        self.vectorizer = CountVectorizer(
            max_features=5000,
            stop_words=list(self.stop_words) if self.language == 'chinese' else 'english'
        )

        # Load sentence transformer model
        try:
            self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        except Exception as e:
            st.error(f"Failed to load Sentence Transformer model: {str(e)}")
            raise
    
    def load_stopwords(self) -> Set[str]:
        """
        Load stopwords.

        Returns:
            Set[str]: Set of stopwords
        """
        try:
            if self.language == 'chinese':
                with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                    return set(line.strip() for line in f)
            else:
                from nltk.corpus import stopwords
                return set(stopwords.words('english'))
        except Exception as e:
            st.error(f"Failed to load stopwords: {str(e)}")
            return set()
    
    @staticmethod
    @st.cache_data
    def _cached_vectorize_texts(texts: List[str], model) -> np.ndarray:
        """
        Cached text vectorization method.

        Args:
            texts: List of texts
            model: SentenceTransformer model instance

        Returns:
            np.ndarray: Text vector matrix
        """
        try:
            embeddings = []
            with st.progress(0) as progress:
                for i, text in enumerate(texts):
                    embedding = model.encode(text)
                    embeddings.append(embedding)
                    progress.progress((i + 1) / len(texts))
            return np.array(embeddings)
        except Exception as e:
            st.error(f"Failed to vectorize texts: {str(e)}")
            return np.array([])
    
    def analyze_topics(self, texts: List[str],
                      n_topics: int = 5,
                      method: str = 'lda') -> Dict:
        """
        Main topic analysis method.

        Args:
            texts: List of review texts
            n_topics: Number of topics
            method: 'lda' or 'kmeans'

        Returns:
            Dict: Topic analysis results
        """
        try:
            # Text vectorization
            if method == 'lda':
                # Use bag-of-words model
                text_vectors = self.vectorizer.fit_transform(texts)
                results = self._run_lda(text_vectors, n_topics)
            else:
                # Use sentence embeddings
                text_vectors = self._cached_vectorize_texts(texts, self.sentence_model)
                results = self._run_kmeans(text_vectors, n_topics, texts)

            # Add example documents
            results['example_docs'] = self._get_topic_examples(
                texts,
                results['document_topics'],
                n_examples=3
            )

            return results

        except Exception as e:
            st.error(f"Failed to analyze topics: {str(e)}")
            return {}
    
    def _run_lda(self, text_vectors: np.ndarray, n_topics: int) -> Dict:
        """
        Run LDA topic model.

        Args:
            text_vectors: Text vector matrix
            n_topics: Number of topics

        Returns:
            Dict: LDA analysis results
        """
        try:
            # Initialize and train LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            doc_topics = self.lda_model.fit_transform(text_vectors)

            # Extract topic keywords
            feature_names = self.vectorizer.get_feature_names_out()
            keywords_per_topic = []
            topic_weights = []

            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_indices = topic.argsort()[:-10-1:-1]
                top_keywords = [feature_names[i] for i in top_indices]
                top_weights = [topic[i] for i in top_indices]

                keywords_per_topic.append(top_keywords)
                topic_weights.append(top_weights)

            return {
                'topics': keywords_per_topic,
                'topic_weights': topic_weights,
                'document_topics': doc_topics.argmax(axis=1),
                'topic_distribution': doc_topics
            }

        except Exception as e:
            st.error(f"LDA analysis failed: {str(e)}")
            return {}
    
    def _run_kmeans(self, text_vectors: np.ndarray, n_topics: int, texts: list = None) -> Dict:
        """
        Run KMeans clustering.

        Args:
            text_vectors: Text vector matrix
            n_topics: Number of clusters

        Returns:
            Dict: Clustering results
        """
        try:
            # Initialize and train KMeans model
            self.kmeans_model = KMeans(
                n_clusters=n_topics,
                random_state=42
            )
            cluster_labels = self.kmeans_model.fit_predict(text_vectors)

            # Extract keywords for each cluster
            keywords_per_cluster = []
            for i in range(n_topics):
                cluster_texts = [text for text, label in zip(texts, cluster_labels) if label == i]
                if cluster_texts:
                    # Extract keywords using TF-IDF
                    keywords = self.extract_keywords(cluster_texts, top_n=10)
                    keywords_per_cluster.append(list(keywords.keys()))
                else:
                    keywords_per_cluster.append([])

            return {
                'topics': keywords_per_cluster,
                'document_topics': cluster_labels,
                'cluster_centers': self.kmeans_model.cluster_centers_
            }

        except Exception as e:
            st.error(f"KMeans clustering failed: {str(e)}")
            return {}
    
    def _get_topic_examples(self, texts: List[str],
                          document_topics: np.ndarray,
                          n_examples: int = 3) -> Dict[int, List[str]]:
        """
        Get example documents for each topic.

        Args:
            texts: List of texts
            document_topics: Document-topic assignments
            n_examples: Number of examples per topic

        Returns:
            Dict[int, List[str]]: Example documents per topic
        """
        examples = {}
        for topic_id in range(len(set(document_topics))):
            topic_docs = [text for text, topic in zip(texts, document_topics) if topic == topic_id]
            examples[topic_id] = topic_docs[:n_examples]
        return examples
    
    def get_topic_trends(self, df: pd.DataFrame,
                        document_topics: List[int],
                        time_window: str = 'M') -> pd.DataFrame:
        """
        Analyze topic trends over time.

        Args:
            df: DataFrame containing timestamps
            document_topics: Document-topic assignments
            time_window: Time window

        Returns:
            pd.DataFrame: Topic trend data
        """
        try:
            # Add topic labels to DataFrame
            df = df.copy()
            df['topic'] = document_topics

            # Count topic distribution by time window
            topic_trends = df.groupby([
                pd.Grouper(key='timestamp', freq=time_window),
                'topic'
            ]).size().unstack(fill_value=0)

            # Calculate proportions
            topic_trends_pct = topic_trends.div(topic_trends.sum(axis=1), axis=0) * 100

            return topic_trends_pct

        except Exception as e:
            st.error(f"Failed to analyze topic trends: {str(e)}")
            return pd.DataFrame()

class InsightAnalyzer(TextAnalyzer):
    """Review insight analyzer."""

    def __init__(self, language: str = 'chinese'):
        """
        Initialize the insight analyzer.

        Args:
            language: Text language
        """
        super().__init__(language)

        # Initialize anomaly detector
        self.outlier_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def analyze_rating_sentiment_correlation(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the correlation between ratings and sentiment.

        Args:
            df: Review data DataFrame

        Returns:
            Dict: Correlation analysis results
        """
        try:
            # Check if required columns exist
            if 'sentiment' not in df.columns or 'rating' not in df.columns:
                return {
                    'correlation': 0,
                    'consistency': 0,
                    'error': 'Missing required columns (sentiment or rating)'
                }

            # Convert sentiment labels to numeric values
            sentiment_map = {'positive': 1, 'neutral': 0.5, 'negative': 0}
            sentiment_scores = df['sentiment'].map(sentiment_map)

            # Check if there are valid sentiment scores
            if sentiment_scores.isna().all():
                return {
                    'correlation': 0,
                    'consistency': 0,
                    'error': 'Unable to convert sentiment labels'
                }

            # Calculate correlation coefficient
            correlation = np.corrcoef(df['rating'], sentiment_scores)[0, 1]

            # Calculate consistency
            consistency = (
                (df['rating'] >= 4) & (df['sentiment'] == 'positive') |
                (df['rating'] <= 2) & (df['sentiment'] == 'negative') |
                (df['rating'] == 3) & (df['sentiment'] == 'neutral')
            ).mean()
            
            return {
                'correlation': correlation,
                'consistency': consistency
            }
            
        except Exception as e:
            st.error(f"Correlation analysis failed: {str(e)}")
            return {
                'correlation': 0,
                'consistency': 0,
                'error': str(e)
            }
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect review anomalies.

        Args:
            df: DataFrame containing review data

        Returns:
            pd.DataFrame: DataFrame with anomaly flags
        """
        try:
            # Copy dataframe to avoid modifying original data
            df_copy = df.copy()

            # Prepare features
            features = []
            feature_names = []

            # 1. Rating feature
            if 'rating' in df_copy.columns:
                features.append(df_copy['rating'])
                feature_names.append('rating')

            # 2. Review length feature
            df_copy['text_length'] = df_copy['content'].str.len()
            features.append(df_copy['text_length'])
            feature_names.append('text_length')

            # 3. Include sentiment score if available
            if 'sentiment_score' in df_copy.columns:
                features.append(df_copy['sentiment_score'])
                feature_names.append('sentiment_score')

            # Combine features into a matrix
            X = np.column_stack(features)

            # Standardize features
            X_scaled = self.scaler.fit_transform(X)

            # Detect anomalies
            anomaly_labels = self.outlier_detector.fit_predict(X_scaled)
            df_copy['is_anomaly'] = anomaly_labels == -1

            # Add anomaly reasons
            df_copy['anomaly_reason'] = ''
            anomaly_mask = df_copy['is_anomaly']

            # Calculate Z-scores for each feature
            for i, feature in enumerate(feature_names):
                z_scores = np.abs(stats.zscore(X_scaled[:, i]))

                # Add specific reasons based on feature type
                if feature == 'rating':
                    mask = (z_scores > 2) & anomaly_mask
                    df_copy.loc[mask, 'anomaly_reason'] += 'abnormal rating '

                elif feature == 'text_length':
                    # Distinguish between abnormally short and long reviews
                    length_mean = df_copy['text_length'].mean()
                    extremely_short = (df_copy['text_length'] < length_mean/2) & anomaly_mask
                    extremely_long = (df_copy['text_length'] > length_mean*2) & anomaly_mask

                    df_copy.loc[extremely_short, 'anomaly_reason'] += 'review too short '
                    df_copy.loc[extremely_long, 'anomaly_reason'] += 'review too long '

                elif feature == 'sentiment_score':
                    mask = (z_scores > 2) & anomaly_mask
                    df_copy.loc[mask, 'anomaly_reason'] += 'abnormal sentiment '

            # Clean up anomaly reason formatting
            df_copy['anomaly_reason'] = df_copy['anomaly_reason'].str.strip()
            df_copy.loc[anomaly_mask & (df_copy['anomaly_reason'] == ''), 'anomaly_reason'] = 'composite feature anomaly'

            return df_copy

        except Exception as e:
            st.error(f"Anomaly detection failed: {str(e)}")
            return df

    
    @staticmethod
    @st.cache_data
    def cached_extract_insights(df: pd.DataFrame, language: str) -> Dict:
        """
        Cached version of the insight extraction method.

        Args:
            df: DataFrame containing review data
            language: Text language

        Returns:
            Dict: Insight results
        """
        analyzer = InsightAnalyzer(language)
        return analyzer._extract_insights(df)

    def extract_insights(self, df: pd.DataFrame) -> Dict:
        """
        Extract review insights.

        Args:
            df: DataFrame containing review data

        Returns:
            Dict: Insight results
        """
        try:
            # Detect anomalies and get DataFrame with anomaly flags
            df_with_anomalies = self.detect_anomalies(df)

            # Analyze correlations
            correlations = self.analyze_rating_sentiment_correlation(df_with_anomalies)

            # Summarize insights
            insights = {
                'anomalies': {
                    'total': int(df_with_anomalies['is_anomaly'].sum()),
                    'details': df_with_anomalies[df_with_anomalies['is_anomaly']],
                    'reasons': df_with_anomalies[df_with_anomalies['is_anomaly']]['anomaly_reason'].value_counts().to_dict(),
                    'df': df_with_anomalies  # Save the full DataFrame with anomaly flags
                },
                'correlations': correlations
            }

            return insights

        except Exception as e:
            st.error(f"Insight analysis failed: {str(e)}")
            return {}

    def _extract_insights(self, df: pd.DataFrame) -> Dict:
        """
        Internal insight extraction logic.

        Args:
            df: DataFrame containing review data

        Returns:
            Dict: Insight results
        """
        try:
            # Detect anomalies
            df_with_anomalies = self.detect_anomalies(df)

            # Analyze correlations
            correlations = self.analyze_rating_sentiment_correlation(df)

            # Summarize insights
            insights = {
                'anomalies': {
                    'total': df_with_anomalies['is_anomaly'].sum(),
                    'details': df_with_anomalies[df_with_anomalies['is_anomaly']],
                    'reasons': df_with_anomalies['anomaly_reason'].value_counts().to_dict()
                },
                'correlations': correlations
            }

            return insights

        except Exception as e:
            st.error(f"Insight analysis failed: {str(e)}")
            return {}