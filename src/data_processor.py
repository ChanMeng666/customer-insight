# import pandas as pd
# import numpy as np
# from datetime import datetime
# from typing import Optional, Dict, List

# class DataProcessor:
#     """
#     数据处理类，负责数据的加载、验证、清洗和统计分析
#     """
    
#     REQUIRED_COLUMNS = ['review_text', 'rating', 'timestamp']
    
#     def __init__(self):
#         self.data: Optional[pd.DataFrame] = None
#         self.stats: Dict = {}
        
#     def load_data(self, file_path: str) -> bool:
#         """
#         加载数据文件并进行初始验证
        
#         Args:
#             file_path: 数据文件路径
            
#         Returns:
#             bool: 是否成功加载数据
#         """
#         try:
#             if file_path.endswith('.csv'):
#                 self.data = pd.read_csv(file_path)
#             elif file_path.endswith('.xlsx'):
#                 self.data = pd.read_excel(file_path)
            
#             # 验证必要字段
#             if not self._validate_columns():
#                 raise ValueError("数据缺少必要字段")
                
#             # 转换时间戳格式
#             self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            
#             return True
#         except Exception as e:
#             print(f"数据加载错误：{str(e)}")
#             return False
    
#     def _validate_columns(self) -> bool:
#         """
#         验证数据是否包含必要的列
        
#         Returns:
#             bool: 验证是否通过
#         """
#         return all(col in self.data.columns for col in self.REQUIRED_COLUMNS)
    
#     def clean_data(self) -> bool:
#         """
#         清洗数据，包括处理空值、异常值等
        
#         Returns:
#             bool: 清洗是否成功
#         """
#         if self.data is None:
#             return False
            
#         try:
#             # 删除评论文本为空的行
#             self.data = self.data.dropna(subset=['review_text'])
            
#             # 处理评分异常值
#             self.data = self.data[self.data['rating'].between(1, 5)]
            
#             # 重置索引
#             self.data = self.data.reset_index(drop=True)
            
#             return True
#         except Exception as e:
#             print(f"数据清洗错误：{str(e)}")
#             return False
    
#     def calculate_statistics(self) -> Dict:
#         """
#         计算基本统计信息
        
#         Returns:
#             Dict: 包含各种统计指标的字典
#         """
#         if self.data is None:
#             return {}
            
#         try:
#             self.stats = {
#                 'total_reviews': len(self.data),
#                 'average_rating': round(self.data['rating'].mean(), 2),
#                 'rating_distribution': self.data['rating'].value_counts().to_dict(),
#                 'daily_reviews': self.data.groupby(
#                     self.data['timestamp'].dt.date
#                 ).size().to_dict()
#             }
            
#             return self.stats
#         except Exception as e:
#             print(f"统计计算错误：{str(e)}")
#             return {}
    
#     def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
#         """
#         按日期范围筛选数据
        
#         Args:
#             start_date: 开始日期
#             end_date: 结束日期
            
#         Returns:
#             pd.DataFrame: 筛选后的数据
#         """
#         if self.data is None:
#             return pd.DataFrame()
            
#         mask = (self.data['timestamp'].dt.date >= start_date.date()) & \
#                (self.data['timestamp'].dt.date <= end_date.date())
#         return self.data[mask]


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import jieba
import jieba.analyse
import os
import streamlit as st

class DataProcessor:
    """
    数据处理类，负责数据的加载、验证、清洗和统计分析
    """
    
    REQUIRED_COLUMNS = ['content', 'rating', 'timestamp']
    
    def __init__(self):
        """
        初始化数据处理器
        """
        self.data: Optional[pd.DataFrame] = None
        self.stats: Dict = {}
        
        # 初始化jieba
        self._initialize_jieba()
    
    def _initialize_jieba(self):
        """
        初始化jieba分词器和自定义词典
        """
        try:
            # 确保jieba被正确初始化
            jieba.initialize()
            
            # 设置停用词文件路径
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            stopwords_path = os.path.join(current_dir, 'utils', 'chinese_stopwords.txt')
            
            # 确保停用词文件存在
            if os.path.exists(stopwords_path):
                jieba.analyse.set_stop_words(stopwords_path)
            else:
                st.warning(f"停用词文件不存在: {stopwords_path}")
            
            # 添加自定义词典
            custom_words = [
                '唐人街', '华裔', 'PYIFF', '映后', '纪录片',
                '平遥', '导演', '影评', '观众', '剧情'
            ]
            for word in custom_words:
                jieba.add_word(word)
                
        except Exception as e:
            st.error(f"jieba初始化失败: {str(e)}")
    
    def load_data(self, file_path: str) -> bool:
        """
        加载数据文件并进行初始验证
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            bool: 是否成功加载数据
        """
        try:
            # 设置数据类型字典
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
                raise ValueError("不支持的文件格式,请使用CSV或Excel文件")
            
            # 验证必要字段
            if not self._validate_columns():
                raise ValueError(f"数据缺少必要字段: {', '.join(self.REQUIRED_COLUMNS)}")
            
            # 处理时间戳
            self._process_timestamp()
            
            # 清理数据
            self.clean_data()
            
            return True
            
        except Exception as e:
            st.error(f"数据加载错误：{str(e)}")
            return False
    
    def _process_timestamp(self):
        """
        处理时间戳列
        """
        try:
            if 'timestamp' in self.data.columns:
                # 尝试转换时间戳
                self.data['timestamp'] = pd.to_datetime(
                    self.data['timestamp'],
                    format='%Y/%m/%d %H:%M',
                    errors='coerce'
                )
                
                # 检查是否有无效的时间戳
                invalid_dates = self.data['timestamp'].isnull().sum()
                if invalid_dates > 0:
                    st.warning(f"发现{invalid_dates}条无效的时间戳记录")
                
                # 删除无效的时间戳
                self.data = self.data.dropna(subset=['timestamp'])
                
        except Exception as e:
            st.error(f"时间戳处理失败: {str(e)}")
    
    def _validate_columns(self) -> bool:
        """
        验证数据是否包含必要的列
        
        Returns:
            bool: 验证是否通过
        """
        return all(col in self.data.columns for col in self.REQUIRED_COLUMNS)
    
    def clean_data(self) -> bool:
        """
        清洗数据，包括处理空值、异常值等
        
        Returns:
            bool: 清洗是否成功
        """
        if self.data is None:
            return False
            
        try:
            # 删除空值
            original_length = len(self.data)
            self.data = self.data.dropna(subset=['content', 'rating'])
            dropped_rows = original_length - len(self.data)
            if dropped_rows > 0:
                st.warning(f"删除了{dropped_rows}条空值记录")
            
            # 处理评分异常值
            self.data = self.data[self.data['rating'].between(1, 5)]
            
            # 处理文本内容
            self.data['content'] = self.data['content'].astype(str).apply(self._clean_text)
            
            # 添加长度列
            self.data['text_length'] = self.data['content'].str.len()
            
            # 如果没有category列，添加一个默认的
            if 'category' not in self.data.columns:
                self.data['category'] = 'default'
            
            # 重置索引
            self.data = self.data.reset_index(drop=True)
            
            return True
            
        except Exception as e:
            st.error(f"数据清洗错误：{str(e)}")
            return False
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 输入文本
            
        Returns:
            str: 清理后的文本
        """
        import re
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        return text.strip()
    
    def calculate_statistics(self) -> Dict:
        """
        计算基本统计信息
        
        Returns:
            Dict: 包含各种统计指标的字典
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
            st.error(f"统计计算错误：{str(e)}")
            return {}
    
    def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        按日期范围筛选数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 筛选后的数据
        """
        if self.data is None:
            return pd.DataFrame()
        
        try:    
            mask = (self.data['timestamp'].dt.date >= start_date.date()) & \
                   (self.data['timestamp'].dt.date <= end_date.date())
            return self.data[mask]
            
        except Exception as e:
            st.error(f"日期筛选错误：{str(e)}")
            return pd.DataFrame()
    
    def filter_by_rating(self, min_rating: float, max_rating: float) -> pd.DataFrame:
        """
        按评分范围筛选数据
        
        Args:
            min_rating: 最小评分
            max_rating: 最大评分
            
        Returns:
            pd.DataFrame: 筛选后的数据
        """
        if self.data is None:
            return pd.DataFrame()
            
        try:
            return self.data[
                (self.data['rating'] >= min_rating) &
                (self.data['rating'] <= max_rating)
            ]
        except Exception as e:
            st.error(f"评分筛选错误：{str(e)}")
            return pd.DataFrame()