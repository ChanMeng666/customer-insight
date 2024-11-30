import jieba
import jieba.analyse
import os
import streamlit as st

def initialize_jieba():
    """初始化jieba配置"""
    # 设置停用词文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_path = os.path.join(current_dir, 'chinese_stopwords.txt')
    
    # 确保停用词文件存在
    if not os.path.exists(stopwords_path):
        raise FileNotFoundError(f"停用词文件不存在: {stopwords_path}")
    
    # 设置jieba的停用词
    jieba.analyse.set_stop_words(stopwords_path)
    
    # 添加自定义词典
    custom_words = ['唐人街', '华裔', 'PYIFF']
    for word in custom_words:
        jieba.add_word(word)
    
    return True