import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-09-01 10:00', '2024-09-02 11:00', '2024-09-03 12:00',
            '2024-09-04 13:00', '2024-09-05 14:00'
        ]),
        'content': [
            'This movie was excellent and very moving',
            'Terrible film, waste of time',
            'Average movie, nothing special',
            'Great cinematography and acting',
            'Boring and predictable plot'
        ],
        'rating': [5.0, 1.0, 3.0, 4.0, 2.0],
        'category': ['positive', 'negative', 'neutral', 'positive', 'negative'],
        'user_id': ['u1', 'u2', 'u3', 'u4', 'u5']
    })


@pytest.fixture
def sample_df_chinese():
    """Create a sample Chinese DataFrame for testing."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-09-01 10:00', '2024-09-02 11:00', '2024-09-03 12:00'
        ]),
        'content': [
            '这部电影非常好看，很感人',
            '很差的电影，浪费时间',
            '一般般的电影，没什么特别的'
        ],
        'rating': [5.0, 1.0, 3.0],
        'category': ['positive', 'negative', 'neutral'],
        'user_id': ['u1', 'u2', 'u3']
    })
