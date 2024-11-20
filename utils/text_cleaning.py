import re
import jieba
import html
from typing import List, Set
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextCleaner:
    """
    文本清洗工具类，提供文本预处理、分词和标准化功能
    """
    
    def __init__(self, language: str = "chinese"):
        """
        初始化文本清洗器
        
        Args:
            language: 文本语言，支持 'chinese' 或 'english'
        """
        self.language = language
        self.stopwords: Set[str] = set()
        self.load_stopwords()
        
    def load_stopwords(self) -> None:
        """
        加载停用词表
        """
        try:
            if self.language == "chinese":
                # 加载中文停用词（这里需要添加中文停用词文件路径）
                with open('utils/chinese_stopwords.txt', 'r', encoding='utf-8') as f:
                    self.stopwords = set(line.strip() for line in f)
            else:
                self.stopwords = set(stopwords.words('english'))
        except Exception as e:
            print(f"停用词加载失败：{str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text: 输入文本
            
        Returns:
            str: 预处理后的文本
        """
        # 转换HTML实体
        text = html.unescape(text)
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除表情符号
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        
        return text.strip()
    
    def segment_text(self, text: str) -> List[str]:
        """
        文本分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果列表
        """
        if self.language == "chinese":
            return list(jieba.cut(text))
        else:
            return word_tokenize(text)
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """
        移除停用词
        
        Args:
            words: 分词列表
            
        Returns:
            List[str]: 移除停用词后的词列表
        """
        return [w for w in words if w not in self.stopwords]
    
    def clean_text(self, text: str) -> str:
        """
        完整的文本清洗流程
        
        Args:
            text: 输入文本
            
        Returns:
            str: 清洗后的文本
        """
        # 预处理
        text = self.preprocess_text(text)
        
        # 分词
        words = self.segment_text(text)
        
        # 移除停用词
        words = self.remove_stopwords(words)
        
        # 标准化
        if self.language != "chinese":
            words = [w.lower() for w in words]
        
        return " ".join(words) 