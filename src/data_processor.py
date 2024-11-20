import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List

class DataProcessor:
    """
    数据处理类，负责数据的加载、验证、清洗和统计分析
    """
    
    REQUIRED_COLUMNS = ['review_text', 'rating', 'timestamp']
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.stats: Dict = {}
        
    def load_data(self, file_path: str) -> bool:
        """
        加载数据文件并进行初始验证
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            bool: 是否成功加载数据
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            
            # 验证必要字段
            if not self._validate_columns():
                raise ValueError("数据缺少必要字段")
                
            # 转换时间戳格式
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            
            return True
        except Exception as e:
            print(f"数据加载错误：{str(e)}")
            return False
    
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
            # 删除评论文本为空的行
            self.data = self.data.dropna(subset=['review_text'])
            
            # 处理评分异常值
            self.data = self.data[self.data['rating'].between(1, 5)]
            
            # 重置索引
            self.data = self.data.reset_index(drop=True)
            
            return True
        except Exception as e:
            print(f"数据清洗错误：{str(e)}")
            return False
    
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
                ).size().to_dict()
            }
            
            return self.stats
        except Exception as e:
            print(f"统计计算错误：{str(e)}")
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
            
        mask = (self.data['timestamp'].dt.date >= start_date.date()) & \
               (self.data['timestamp'].dt.date <= end_date.date())
        return self.data[mask]