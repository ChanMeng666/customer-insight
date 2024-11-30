import os
from setuptools import setup, find_packages

def read_requirements():
    """读取requirements.txt文件"""
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def ensure_dirs():
    """确保必要的目录存在"""
    dirs = ['utils']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def ensure_stopwords():
    """确保停用词文件存在"""
    stopwords_path = 'utils/chinese_stopwords.txt'
    if not os.path.exists(stopwords_path):
        with open(stopwords_path, 'w', encoding='utf-8') as f:
            # 这里写入基本的停用词
            f.write('的\n了\n和\n是\n就\n都\n而\n及\n与\n着\n或\n一个\n没有\n我们\n你们\n他们\n')

if __name__ == '__main__':
    ensure_dirs()
    ensure_stopwords()
    
    setup(
        name='comment_analysis',
        version='1.0',
        packages=find_packages(),
        install_requires=read_requirements(),
        python_requires='>=3.7',
    )