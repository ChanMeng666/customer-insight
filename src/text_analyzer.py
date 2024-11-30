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

# class KeywordAnalyzer(TextAnalyzer):
#     """关键词分析器"""
    
#     def __init__(self, language: str = 'chinese'):
#         """
#         初始化关键词分析器
        
#         Args:
#             language: 文本语言，支持 'chinese' 或 'english'
#         """
#         super().__init__(language)
#         self.stop_words = load_stopwords(language)
        
#         # 配置TF-IDF分析器
#         self.tfidf = TfidfVectorizer(
#             max_features=1000,
#             stop_words=self.stop_words,
#             tokenizer=self._tokenize_text
#         )
    
#     def _tokenize_text(self, text: str) -> List[str]:
#         """
#         文本分词
        
#         Args:
#             text: 输入文本
            
#         Returns:
#             List[str]: 分词结果
#         """
#         if self.language == 'chinese':
#             return list(jieba.cut(text))
#         else:
#             from nltk.tokenize import word_tokenize
#             return word_tokenize(text.lower())
    
#     @st.cache_data
#     def extract_keywords(self, texts: List[str], top_n: int = 20) -> Dict[str, float]:
#         """
#         提取关键词及其权重
        
#         Args:
#             texts: 评论文本列表
#             top_n: 返回前N个关键词
            
#         Returns:
#             Dict[str, float]: 关键词及其权重
#         """
#         try:
#             if self.language == 'chinese':
#                 # 使用jieba的TF-IDF实现
#                 keywords = {}
#                 for text in texts:
#                     for kw, weight in jieba.analyse.extract_tags(text, topK=top_n, withWeight=True):
#                         if kw in keywords:
#                             keywords[kw] += weight
#                         else:
#                             keywords[kw] = weight
#             else:
#                 # 使用sklearn的TF-IDF实现
#                 tfidf_matrix = self.tfidf.fit_transform(texts)
#                 feature_names = self.tfidf.get_feature_names_out()
                
#                 # 计算每个词的平均TF-IDF值
#                 keywords = {}
#                 for idx, word in enumerate(feature_names):
#                     weight = np.mean(tfidf_matrix[:, idx].toarray())
#                     keywords[word] = weight
            
#             # 排序并返回前N个关键词
#             sorted_keywords = dict(sorted(keywords.items(), 
#                                        key=lambda x: x[1], 
#                                        reverse=True)[:top_n])
#             return sorted_keywords
            
#         except Exception as e:
#             st.error(f"关键词提取失败：{str(e)}")
#             return {}
    
#     @st.cache_data
#     def extract_keywords_by_rating(self, df: pd.DataFrame, top_n: int = 20) -> Dict[str, Dict[str, float]]:
#         """
#         按评分分类提取关键词
        
#         Args:
#             df: 包含评论文本和评分的DataFrame
#             top_n: 每类返回前N个关键词
            
#         Returns:
#             Dict: 各评分段的关键词及权重
#         """
#         try:
#             # 将评分分为高分低分
#             high_ratings = df[df['rating'] >= 4]['review_text'].tolist()
#             low_ratings = df[df['rating'] <= 2]['review_text'].tolist()
            
#             # 分别提取关键词
#             positive_keywords = self.extract_keywords(high_ratings, top_n)
#             negative_keywords = self.extract_keywords(low_ratings, top_n)
            
#             return {
#                 'positive': positive_keywords,
#                 'negative': negative_keywords
#             }
            
#         except Exception as e:
#             st.error(f"评分关键词分析失败：{str(e)}")
#             return {'positive': {}, 'negative': {}}
    
#     @st.cache_data
#     def calculate_keyword_trends(self, df: pd.DataFrame, 
#                                top_keywords: List[str], 
#                                time_window: str = 'M') -> pd.DataFrame:
#         """
#         计算关键词随时间的变化趋势
        
#         Args:
#             df: 包含评论文本和时间戳的DataFrame
#             top_keywords: 要追踪的关键词列表
#             time_window: 时间窗口 ('D'=日, 'W'=周, 'M'=月)
            
#         Returns:
#             pd.DataFrame: 关键词趋势数据
#         """
#         try:
#             # 按时间窗口分组
#             grouped = df.groupby(pd.Grouper(key='timestamp', freq=time_window))
            
#             # 初始化结果DataFrame
#             trend_data = []
            
#             # 计算每个时间窗口的关键词频率
#             for time, group in grouped:
#                 if len(group) == 0:
#                     continue
                    
#                 # 提取该时间窗口的关键词
#                 texts = group['review_text'].tolist()
#                 keywords = self.extract_keywords(texts, len(top_keywords))
                
#                 # 记录每个关注关键词频率
#                 for keyword in top_keywords:
#                     trend_data.append({
#                         'timestamp': time,
#                         'keyword': keyword,
#                         'frequency': keywords.get(keyword, 0)
#                     })
            
#             return pd.DataFrame(trend_data)
            
#         except Exception as e:
#             st.error(f"关键词趋势分析失败：{str(e)}")
#             return pd.DataFrame()


# 修改text_analyzer.py中的KeywordAnalyzer类

class KeywordAnalyzer(TextAnalyzer):
    def __init__(self, language: str = 'chinese'):
        super().__init__(language)
        self.stop_words = load_stopwords(language)
        
        # 添加评论相关的停用词
        self.stop_words.update({'电影', '片子', '导演', '拍摄', '#', 'PYIFF'})
        
        # 配置jieba分词
        import jieba.analyse
        jieba.analyse.set_stop_words('utils/chinese_stopwords.txt')
        
        # 自定义词典
        for word in ['唐人街', '华裔']:
            jieba.add_word(word)
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 移除#号标签
        text = re.sub(r'#.*?#', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 移除URL
        text = re.sub(r'http[s]?://\S+', '', text)
        return text.strip()
    
    def extract_keywords(self, texts: List[str], top_n: int = 20) -> Dict[str, float]:
        """提取关键词及其权重"""
        try:
            # 预处理文本
            processed_texts = [self._preprocess_text(text) for text in texts]
            combined_text = ' '.join(processed_texts)
            
            # 使用jieba提取关键词
            keywords = {}
            for kw, weight in jieba.analyse.extract_tags(
                combined_text,
                topK=top_n,
                withWeight=True,
                allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn')  # 允许更多词性
            ):
                if kw not in self.stop_words and len(kw) > 1:  # 过滤单字词
                    keywords[kw] = float(weight)
            
            # 标准化权重
            if keywords:
                max_weight = max(keywords.values())
                keywords = {k: float(v/max_weight) for k, v in keywords.items()}
            
            return keywords
            
        except Exception as e:
            st.error(f"关键词提取失败: {str(e)}")
            return {}
    
    def calculate_keyword_trends(self, df: pd.DataFrame, 
                               top_keywords: List[str], 
                               time_window: str = 'M') -> pd.DataFrame:
        """计算关键词趋势"""
        try:
            df = df.copy()
            
            # 确保timestamp列是datetime类型
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M')
            
            # 按时间窗口和category分组
            grouped = df.groupby([
                pd.Grouper(key='timestamp', freq=time_window),
                'category'
            ])
            
            trend_data = []
            
            for (time, category), group in grouped:
                if len(group) == 0:
                    continue
                
                # 提取关键词
                texts = group['content'].tolist()
                keywords = self.extract_keywords(texts, len(top_keywords))
                
                # 记录频率
                for keyword in top_keywords:
                    trend_data.append({
                        'timestamp': time,
                        'category': category,
                        'keyword': keyword,
                        'frequency': keywords.get(keyword, 0.0)
                    })
            
            return pd.DataFrame(trend_data)
            
        except Exception as e:
            st.error(f"趋势分析失败: {str(e)}")
            return pd.DataFrame()
    
    def extract_keywords_by_rating(self, df: pd.DataFrame, top_n: int = 20) -> Dict[str, Dict[str, float]]:
        """按评分分类提取关键词"""
        try:
            df = df.copy()
            
            # 根据评分和情感类别分组
            positive_mask = (df['rating'] >= 4) | (df['category'].str.contains('positive', case=False))
            negative_mask = (df['rating'] <= 2) | (df['category'].str.contains('negative', case=False))
            
            positive_texts = df[positive_mask]['content'].tolist()
            negative_texts = df[negative_mask]['content'].tolist()
            
            # 提取关键词
            positive_keywords = self.extract_keywords(positive_texts, top_n) if positive_texts else {}
            negative_keywords = self.extract_keywords(negative_texts, top_n) if negative_texts else {}
            
            return {
                'positive': positive_keywords,
                'negative': negative_keywords
            }
            
        except Exception as e:
            st.error(f"评分关键词分析失败: {str(e)}")
            return {'positive': {}, 'negative': {}}

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
        self.stop_words = self.load_stopwords()
        
        # 初始化向量化器
        self.vectorizer = CountVectorizer(
            max_features=5000,
            stop_words=list(self.stop_words) if self.language == 'chinese' else 'english'
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
    
    @staticmethod
    @st.cache_data
    def _cached_vectorize_texts(texts: List[str], model) -> np.ndarray:
        """
        缓存版本的文本向量化方法
        
        Args:
            texts: 文本列表
            model: SentenceTransformer模型实例
            
        Returns:
            np.ndarray: 文本向量矩阵
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
            st.error(f"文本向量化失败：{str(e)}")
            return np.array([])
    
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
                text_vectors = self._cached_vectorize_texts(texts, self.sentence_model)
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
        
        # 初始化异常检测器
        self.outlier_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def analyze_rating_sentiment_correlation(self, df: pd.DataFrame) -> Dict:
        """
        分析评分与情感的相关性
        
        Args:
            df: 评论数据DataFrame
            
        Returns:
            Dict: 相关性分析结果
        """
        try:
            # 检查必要的列是否存在
            if 'sentiment' not in df.columns or 'rating' not in df.columns:
                return {
                    'correlation': 0,
                    'consistency': 0,
                    'error': '缺少必要的列（sentiment或rating）'
                }
            
            # 转换情感标签为数值
            sentiment_map = {'正面': 1, '中性': 0.5, '负面': 0}
            sentiment_scores = df['sentiment'].map(sentiment_map)
            
            # 检查是否有有效的情感得分
            if sentiment_scores.isna().all():
                return {
                    'correlation': 0,
                    'consistency': 0,
                    'error': '无法转换情感标签'
                }
            
            # 计算相关系数
            correlation = np.corrcoef(df['rating'], sentiment_scores)[0, 1]
            
            # 计算一致性
            consistency = (
                (df['rating'] >= 4) & (df['sentiment'] == '正面') |
                (df['rating'] <= 2) & (df['sentiment'] == '负面') |
                (df['rating'] == 3) & (df['sentiment'] == '中性')
            ).mean()
            
            return {
                'correlation': correlation,
                'consistency': consistency
            }
            
        except Exception as e:
            st.error(f"相关性分析失败：{str(e)}")
            return {
                'correlation': 0,
                'consistency': 0,
                'error': str(e)
            }
    
    # def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     检测评论异常
        
    #     Args:
    #         df: 包含评论数据的DataFrame
            
    #     Returns:
    #         pd.DataFrame: 带有异常标记的DataFrame
    #     """
    #     try:
    #         # 复制数据框
    #         df_copy = df.copy()
            
    #         # 准备特征
    #         features = []
            
    #         # 1. 评分特征
    #         if 'rating' in df_copy.columns:
    #             features.append(df_copy['rating'])
            
    #         # 2. 评论长度特征
    #         if 'content' in df_copy.columns:
    #             df_copy['review_length'] = df_copy['content'].str.len()
    #             features.append(df_copy['review_length'])
    #         elif 'review_text' in df_copy.columns:  # 添加对 review_text 列的支持
    #             df_copy['review_length'] = df_copy['review_text'].str.len()
    #             features.append(df_copy['review_length'])
            
    #         # 如果没有足够的特征，返回原始数据框
    #         if len(features) < 1:
    #             st.warning("没有足够的特征用于异常检测")
    #             df_copy['is_anomaly'] = False
    #             df_copy['anomaly_reason'] = ''
    #             return df_copy
            
    #         # 将特征组合成矩阵
    #         X = np.column_stack(features)
            
    #         # 标准化特征
    #         X_scaled = self.scaler.fit_transform(X)
            
    #         # 检测异常
    #         anomaly_labels = self.outlier_detector.fit_predict(X_scaled)
    #         df_copy['is_anomaly'] = anomaly_labels == -1
            
    #         # 添加异常原因
    #         df_copy['anomaly_reason'] = ''
    #         anomaly_mask = df_copy['is_anomaly']
            
    #         # 检测异常长度的评论
    #         if 'review_length' in df_copy.columns:
    #             length_mean = df_copy['review_length'].mean()
    #             length_std = df_copy['review_length'].std()
    #             extremely_short = df_copy['review_length'] < (length_mean - 2 * length_std)
    #             extremely_long = df_copy['review_length'] > (length_mean + 2 * length_std)
                
    #             df_copy.loc[anomaly_mask & extremely_short, 'anomaly_reason'] = '评论异常简短'
    #             df_copy.loc[anomaly_mask & extremely_long, 'anomaly_reason'] = '评论异常冗长'
            
    #         # 对于没有具体原因的异常，标记为"其他异常"
    #         df_copy.loc[anomaly_mask & (df_copy['anomaly_reason'] == ''), 'anomaly_reason'] = '其他异常'
            
    #         return df_copy
            
    #     except Exception as e:
    #         st.error(f"异常检测失败：{str(e)}")
    #         return df


    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测评论异常
        
        Args:
            df: 包含评论数据的DataFrame
            
        Returns:
            pd.DataFrame: 带有异常标记的DataFrame
        """
        try:
            # 复制数据框避免修改原始数据
            df_copy = df.copy()
            
            # 准备特征
            features = []
            feature_names = []
            
            # 1. 评分特征
            if 'rating' in df_copy.columns:
                features.append(df_copy['rating'])
                feature_names.append('rating')
            
            # 2. 评论长度特征
            df_copy['text_length'] = df_copy['content'].str.len()
            features.append(df_copy['text_length'])
            feature_names.append('text_length')
            
            # 3. 如果有情感分数，也加入特征
            if 'sentiment_score' in df_copy.columns:
                features.append(df_copy['sentiment_score'])
                feature_names.append('sentiment_score')
            
            # 将特征组合成矩阵
            X = np.column_stack(features)
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 检测异常
            anomaly_labels = self.outlier_detector.fit_predict(X_scaled)
            df_copy['is_anomaly'] = anomaly_labels == -1
            
            # 添加异常原因
            df_copy['anomaly_reason'] = ''
            anomaly_mask = df_copy['is_anomaly']
            
            # 计算各特征的Z分数
            for i, feature in enumerate(feature_names):
                z_scores = np.abs(stats.zscore(X_scaled[:, i]))
                
                # 根据特征类型添加具体原因
                if feature == 'rating':
                    mask = (z_scores > 2) & anomaly_mask
                    df_copy.loc[mask, 'anomaly_reason'] += '评分异常 '
                
                elif feature == 'text_length':
                    # 区分异常长和异常短
                    length_mean = df_copy['text_length'].mean()
                    extremely_short = (df_copy['text_length'] < length_mean/2) & anomaly_mask
                    extremely_long = (df_copy['text_length'] > length_mean*2) & anomaly_mask
                    
                    df_copy.loc[extremely_short, 'anomaly_reason'] += '评论过短 '
                    df_copy.loc[extremely_long, 'anomaly_reason'] += '评论过长 '
                
                elif feature == 'sentiment_score':
                    mask = (z_scores > 2) & anomaly_mask
                    df_copy.loc[mask, 'anomaly_reason'] += '情感异常 '
            
            # 清理异常原因格式
            df_copy['anomaly_reason'] = df_copy['anomaly_reason'].str.strip()
            df_copy.loc[anomaly_mask & (df_copy['anomaly_reason'] == ''), 'anomaly_reason'] = '综合特征异常'
            
            return df_copy
            
        except Exception as e:
            st.error(f"异常检测失败: {str(e)}")
            return df

    
    # 移除实例方法上的缓存装饰器，改为使用静态方法
    @staticmethod
    @st.cache_data
    def cached_extract_insights(df: pd.DataFrame, language: str) -> Dict:
        """
        缓存版本的洞察提取方法
        
        Args:
            df: 包含评论数据的DataFrame
            language: 文本语言
            
        Returns:
            Dict: 洞察结果
        """
        analyzer = InsightAnalyzer(language)
        return analyzer._extract_insights(df)
    
    # def extract_insights(self, df: pd.DataFrame) -> Dict:
    #     """
    #     提取评论洞察的公共接口
        
    #     Args:
    #         df: 包含评论数据的DataFrame
            
    #     Returns:
    #         Dict: 洞察结果
    #     """
    #     return self.cached_extract_insights(df, self.language)

    def extract_insights(self, df: pd.DataFrame) -> Dict:
        """
        实际的洞察提取逻辑
        
        Args:
            df: 包含评论数据的DataFrame
            
        Returns:
            Dict: 洞察结果
        """
        try:
            # 检测异常并获取带有异常标记的DataFrame
            df_with_anomalies = self.detect_anomalies(df)
            
            # 分析相关性
            correlations = self.analyze_rating_sentiment_correlation(df_with_anomalies)
            
            # 汇总洞察
            insights = {
                'anomalies': {
                    'total': int(df_with_anomalies['is_anomaly'].sum()),
                    'details': df_with_anomalies[df_with_anomalies['is_anomaly']],
                    'reasons': df_with_anomalies[df_with_anomalies['is_anomaly']]['anomaly_reason'].value_counts().to_dict(),
                    'df': df_with_anomalies  # 添加这行，保存完整的带异常标记的DataFrame
                },
                'correlations': correlations
            }
            
            return insights
            
        except Exception as e:
            st.error(f"洞察分析失败：{str(e)}")
            return {}
    
    def _extract_insights(self, df: pd.DataFrame) -> Dict:
        """
        实际的洞察提取逻辑
        
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