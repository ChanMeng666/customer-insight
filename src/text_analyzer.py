from typing import List, Dict, Optional
import torch
from torch.nn.functional import softmax
import streamlit as st
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
    nltk.download('punkt')  # 为word_tokenize下载必要的资源

@st.cache_data
def load_stopwords(language: str) -> Set[str]:
    """
    缓存加载停用词表
    
    Args:
        language: 文本语言
        
    Returns:
        Set[str]: 停用词集合
    """
    try:
        if language == 'chinese':
            with open('utils/chinese_stopwords.txt', 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        else:
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
    except Exception as e:
        st.warning(f"停用词加载失败：{str(e)}")
        return set()

class TextAnalyzer:
    """文本分析基类"""
    
    def __init__(self, language: str = 'chinese'):
        """
        初始化文本分析器
        
        Args:
            language: 文本语言，支持 'chinese' 或 'english'
        """
        self.language = language

class SentimentAnalyzer(TextAnalyzer):
    """文本情感分析器"""
    
    def __init__(self, language: str = 'chinese'):
        """
        初始化情感分析器
        
        Args:
            language: 文本语言，支持 'chinese' 或 'english'
        """
        super().__init__(language)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 选择模型
        self.model_name = 'uer/roberta-base-finetuned-jd-binary-chinese' if language == 'chinese' \
                         else 'nlptown/bert-base-multilingual-uncased-sentiment'
        
        # 加载模型和分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # 设置为评估模式
        except Exception as e:
            st.error(f"模型加载失败：{str(e)}")
            raise
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        批量分析文本情感
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            List[Dict]: 情感分析结果列表
        """
        results = []
        
        try:
            # 分批处理
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
                
                # 预测
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = softmax(outputs.logits, dim=-1)
                
                # 处理结果
                for j, (text, pred) in enumerate(zip(batch_texts, predictions)):
                    if self.language == 'chinese':
                        # 中文模型：二分类（正面/负面）
                        sentiment = '正面' if pred[1] > pred[0] else '负面'
                        confidence = float(pred.max())
                    else:
                        # 英文模型：5分类
                        sentiment_score = int(torch.argmax(pred)) + 1
                        sentiment = self._get_sentiment_label(sentiment_score)
                        confidence = float(pred.max())
                    
                    results.append({
                        'text': text,
                        'sentiment': sentiment,
                        'confidence': confidence
                    })
                
                # 更新进度
                if hasattr(st, 'progress_bar'):
                    progress = (i + len(batch_texts)) / len(texts)
                    st.progress_bar.progress(progress)
                    
        except Exception as e:
            st.error(f"情感分析过程出错：{str(e)}")
            raise
            
        return results
    
    def _get_sentiment_label(self, score: int) -> str:
        """
        将评分转换为情感标签
        
        Args:
            score: 1-5的评分
            
        Returns:
            str: 情感标签
        """
        if score <= 2:
            return '负面'
        elif score == 3:
            return '中性'
        else:
            return '正面'
    
    def get_sentiment_stats(self, results: List[Dict]) -> Dict:
        """
        计算情感分析统计信息
        
        Args:
            results: 情感分析结果列表
            
        Returns:
            Dict: 统计信息
        """
        df = pd.DataFrame(results)
        
        stats = {
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'average_confidence': float(df['confidence'].mean()),
            'typical_positive': df[df['sentiment'] == '正面'].nlargest(3, 'confidence')[['text', 'confidence']].to_dict('records'),
            'typical_negative': df[df['sentiment'] == '负面'].nlargest(3, 'confidence')[['text', 'confidence']].to_dict('records')
        }
        
        return stats
    
    @staticmethod
    @st.cache_data
    def cached_analyze_batch(texts: List[str], model_name: str, device: str, language: str, batch_size: int = 32) -> List[Dict]:
        """
        带缓存的批量文本情感分析
        
        Args:
            texts: 文本列表
            model_name: 模型名称
            device: 设备类型
            language: 文本语言
            batch_size: 批处理大小
            
        Returns:
            List[Dict]: 情感分析结果列表
        """
        analyzer = SentimentAnalyzer(language)
        return analyzer.analyze_batch(texts, batch_size)

class KeywordAnalyzer(TextAnalyzer):
    """关键词分析器"""
    
    def __init__(self, language: str = 'chinese'):
        """
        初始化关键词分析器
        
        Args:
            language: 文本语言，支持 'chinese' 或 'english'
        """
        super().__init__(language)
        self.stop_words = load_stopwords(language)
        
        # 配置TF-IDF分析器
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words=self.stop_words,
            tokenizer=self._tokenize_text
        )
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        文本分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        if self.language == 'chinese':
            return list(jieba.cut(text))
        else:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text.lower())
    
    @st.cache_data
    def extract_keywords(self, texts: List[str], top_n: int = 20) -> Dict[str, float]:
        """
        提取关键词及其权重
        
        Args:
            texts: 评论文本列表
            top_n: 返回前N个关键词
            
        Returns:
            Dict[str, float]: 关键词及其权重
        """
        try:
            if self.language == 'chinese':
                # 使用jieba的TF-IDF实现
                keywords = {}
                for text in texts:
                    for kw, weight in jieba.analyse.extract_tags(text, topK=top_n, withWeight=True):
                        if kw in keywords:
                            keywords[kw] += weight
                        else:
                            keywords[kw] = weight
            else:
                # 使用sklearn的TF-IDF实现
                tfidf_matrix = self.tfidf.fit_transform(texts)
                feature_names = self.tfidf.get_feature_names_out()
                
                # 计算每个词的平均TF-IDF值
                keywords = {}
                for idx, word in enumerate(feature_names):
                    weight = np.mean(tfidf_matrix[:, idx].toarray())
                    keywords[word] = weight
            
            # 排序并返回前N个关键词
            sorted_keywords = dict(sorted(keywords.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:top_n])
            return sorted_keywords
            
        except Exception as e:
            st.error(f"关键词提取失败：{str(e)}")
            return {}
    
    @st.cache_data
    def extract_keywords_by_rating(self, df: pd.DataFrame, top_n: int = 20) -> Dict[str, Dict[str, float]]:
        """
        按评分分类提取关键词
        
        Args:
            df: 包含评论文本和评分的DataFrame
            top_n: 每类返回前N个关键词
            
        Returns:
            Dict: 各评分段的关键词及权重
        """
        try:
            # 将评分分为高分低分
            high_ratings = df[df['rating'] >= 4]['review_text'].tolist()
            low_ratings = df[df['rating'] <= 2]['review_text'].tolist()
            
            # 分别提取关键词
            positive_keywords = self.extract_keywords(high_ratings, top_n)
            negative_keywords = self.extract_keywords(low_ratings, top_n)
            
            return {
                'positive': positive_keywords,
                'negative': negative_keywords
            }
            
        except Exception as e:
            st.error(f"评分关键词分析失败：{str(e)}")
            return {'positive': {}, 'negative': {}}
    
    @st.cache_data
    def calculate_keyword_trends(self, df: pd.DataFrame, 
                               top_keywords: List[str], 
                               time_window: str = 'M') -> pd.DataFrame:
        """
        计算关键词随时间的变化趋势
        
        Args:
            df: 包含评论文本和时间戳的DataFrame
            top_keywords: 要追踪的关键词列表
            time_window: 时间窗口 ('D'=日, 'W'=周, 'M'=月)
            
        Returns:
            pd.DataFrame: 关键词趋势数据
        """
        try:
            # 按时间窗口分组
            grouped = df.groupby(pd.Grouper(key='timestamp', freq=time_window))
            
            # 初始化结果DataFrame
            trend_data = []
            
            # 计算每个时间窗口的关键词频率
            for time, group in grouped:
                if len(group) == 0:
                    continue
                    
                # 提取该时间窗口的关键词
                texts = group['review_text'].tolist()
                keywords = self.extract_keywords(texts, len(top_keywords))
                
                # 记录每个关注关键词频率
                for keyword in top_keywords:
                    trend_data.append({
                        'timestamp': time,
                        'keyword': keyword,
                        'frequency': keywords.get(keyword, 0)
                    })
            
            return pd.DataFrame(trend_data)
            
        except Exception as e:
            st.error(f"关键词趋势分析失败：{str(e)}")
            return pd.DataFrame()

class TopicAnalyzer(TextAnalyzer):
    """主题分析器"""
    
    def __init__(self, language: str = 'chinese'):
        """
        初始化主题分析器
        
        Args:
            language: 文本语言，支持 'chinese' 或 'english'
        """
        super().__init__(language)
        
        # 初始化模型
        self.lda_model = None
        self.kmeans_model = None
        
        # 加载停用词
        self.stop_words = self.load_stopwords()  # 使用父类的方法
        
        # 初始化向量化器
        self.vectorizer = CountVectorizer(
            max_features=5000,
            stop_words=self.stop_words
        )
        
        # 加载sentence transformer模型
        try:
            self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        except Exception as e:
            st.error(f"Sentence Transformer模型加载失败：{str(e)}")
            raise
    
    def load_stopwords(self) -> Set[str]:
        """
        加载停用词
        
        Returns:
            Set[str]: 停用词集合
        """
        try:
            if self.language == 'chinese':
                with open('utils/chinese_stopwords.txt', 'r', encoding='utf-8') as f:
                    return set(line.strip() for line in f)
            else:
                from nltk.corpus import stopwords
                return set(stopwords.words('english'))
        except Exception as e:
            st.error(f"停用词加载失败：{str(e)}")
            return set()
    
    @st.cache_data(show_spinner=False)
    def vectorize_texts(self, texts: List[str]) -> np.ndarray:
        """
        使用sentence-transformers进行文本向量化
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: 文本向量矩阵
        """
        try:
            embeddings = []
            with st.progress(0) as progress:
                for i, text in enumerate(texts):
                    embedding = self.sentence_model.encode(text)
                    embeddings.append(embedding)
                    progress.progress((i + 1) / len(texts))
            return np.array(embeddings)
        except Exception as e:
            st.error(f"文本向量化失败：{str(e)}")
            return np.array([])
    
    @st.cache_data
    def analyze_topics(self, texts: List[str], 
                      n_topics: int = 5,
                      method: str = 'lda') -> Dict:
        """
        主题分析主方法
        
        Args:
            texts: 评论文本列表
            n_topics: 主题数量
            method: 'lda' 或 'kmeans'
            
        Returns:
            Dict: 主题分析结果
        """
        try:
            # 文本向量化
            if method == 'lda':
                # 使用词袋模型
                text_vectors = self.vectorizer.fit_transform(texts)
                results = self._run_lda(text_vectors, n_topics)
            else:
                # 使用sentence embeddings
                text_vectors = self.vectorize_texts(texts)
                results = self._run_kmeans(text_vectors, n_topics)
            
            # 添加示例文档
            results['example_docs'] = self._get_topic_examples(
                texts, 
                results['document_topics'],
                n_examples=3
            )
            
            return results
            
        except Exception as e:
            st.error(f"主题分析失败：{str(e)}")
            return {}
    
    def _run_lda(self, text_vectors: np.ndarray, n_topics: int) -> Dict:
        """
        运行LDA主题模型
        
        Args:
            text_vectors: 文本向量矩阵
            n_topics: 主题数量
            
        Returns:
            Dict: LDA分析结果
        """
        try:
            # 初始化并训练LDA模型
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            doc_topics = self.lda_model.fit_transform(text_vectors)
            
            # 提取主题关键词
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
            st.error(f"LDA分析失败：{str(e)}")
            return {}
    
    def _run_kmeans(self, text_vectors: np.ndarray, n_topics: int) -> Dict:
        """
        运行KMeans聚类
        
        Args:
            text_vectors: 文本向量矩阵
            n_topics: 聚类数量
            
        Returns:
            Dict: 聚类结果
        """
        try:
            # 初始化并训练KMeans模型
            self.kmeans_model = KMeans(
                n_clusters=n_topics,
                random_state=42
            )
            cluster_labels = self.kmeans_model.fit_predict(text_vectors)
            
            # 为每个聚类提取关键词
            keywords_per_cluster = []
            for i in range(n_topics):
                cluster_texts = [text for text, label in zip(texts, cluster_labels) if label == i]
                if cluster_texts:
                    # 使用TF-IDF提取关键词
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
            st.error(f"KMeans聚类失败：{str(e)}")
            return {}
    
    def _get_topic_examples(self, texts: List[str], 
                          document_topics: np.ndarray,
                          n_examples: int = 3) -> Dict[int, List[str]]:
        """
        获取每个主题的示例文档
        
        Args:
            texts: 文本列表
            document_topics: 文档-主题分配
            n_examples: 每个主题的示例数量
            
        Returns:
            Dict[int, List[str]]: 主题示例文档
        """
        examples = {}
        for topic_id in range(len(set(document_topics))):
            topic_docs = [text for text, topic in zip(texts, document_topics) if topic == topic_id]
            examples[topic_id] = topic_docs[:n_examples]
        return examples
    
    @st.cache_data
    def get_topic_trends(self, df: pd.DataFrame, 
                        document_topics: List[int],
                        time_window: str = 'M') -> pd.DataFrame:
        """
        分析主题随时间的变化趋势
        
        Args:
            df: 包含时间戳的DataFrame
            document_topics: 文档-主题分配
            time_window: 时间窗口
            
        Returns:
            pd.DataFrame: 主题趋势数据
        """
        try:
            # 添加主题标签到DataFrame
            df = df.copy()
            df['topic'] = document_topics
            
            # 按时间窗口统计主题分布
            topic_trends = df.groupby([
                pd.Grouper(key='timestamp', freq=time_window),
                'topic'
            ]).size().unstack(fill_value=0)
            
            # 计算比例
            topic_trends_pct = topic_trends.div(topic_trends.sum(axis=1), axis=0) * 100
            
            return topic_trends_pct
            
        except Exception as e:
            st.error(f"主题趋势分析失败：{str(e)}")
            return pd.DataFrame()

class InsightAnalyzer(TextAnalyzer):
    """评论洞察分析器"""
    
    def __init__(self, language: str = 'chinese'):
        """
        初始化洞察分析器
        
        Args:
            language: 文本语言
        """
        super().__init__(language)
        
        # 初始化异常检器
        self.outlier_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    @st.cache_data
    def extract_insights(self, df: pd.DataFrame) -> Dict:
        """
        提取评论洞察
        
        Args:
            df: 包含评论数据的DataFrame
            
        Returns:
            Dict: 洞察结果
        """
        try:
            # 检测异常
            df_with_anomalies = self.detect_anomalies(df)
            
            # 分析相关性
            correlations = self.analyze_rating_sentiment_correlation(df)
            
            # 汇总洞察
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
            st.error(f"洞察分析失败：{str(e)}")
            return {}

    def analyze_rating_sentiment_correlation(self, df: pd.DataFrame) -> Dict:
        """
        分析评分与情感的相关性
        
        Args:
            df: 评论数据DataFrame
            
        Returns:
            Dict: 相关性分析结果
        """
        try:
            # 转换情感签为数值
            sentiment_scores = df['sentiment'].map({'正面': 1, '负面': 0, '中性': 0.5})
            
            # 计算相关系数
            correlation = stats.pearsonr(df['rating'], sentiment_scores)[0]
            
            # 分析一致性
            consistency = (
                (df['rating'] >= 4) & (df['sentiment'] == '正面') |
                (df['rating'] <= 2) & (df['sentiment'] == '负面') |
                (df['rating'] == 3) & (df['sentiment'] == '中性')
            ).mean()
            
            return {
                'correlation': correlation,
                'consistency': consistency,
                'inconsistent_examples': self._get_inconsistent_examples(df)
            }
            
        except Exception as e:
            st.error(f"相关性分析失败：{str(e)}")
            return {}

    def _get_inconsistent_examples(self, df: pd.DataFrame) -> List[Dict]:
        """获取情感-评分不一致的例子"""
        inconsistent = df[
            ((df['rating'] >= 4) & (df['sentiment'] == '负面')) |
            ((df['rating'] <= 2) & (df['sentiment'] == '正面'))
        ]
        
        return inconsistent[['review_text', 'rating', 'sentiment']].head(5).to_dict('records')