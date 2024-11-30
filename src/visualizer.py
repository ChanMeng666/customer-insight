import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import networkx as nx
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st

class Visualizer:
    """基础可视化类"""
    
    def __init__(self):
        """初始化基础可视化器"""
        pass
        
    def create_line_plot(self, df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """
        创建折线图
        
        Args:
            df: 数据框
            x: x轴列名
            y: y轴列名
            title: 图表标题
            
        Returns:
            go.Figure: Plotly图表对象
        """
        fig = px.line(df, x=x, y=y, title=title)
        return fig
        
    def create_bar_plot(self, df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """
        创建柱状图
        
        Args:
            df: 数据框
            x: x轴列名
            y: y轴列名
            title: 图表标题
            
        Returns:
            go.Figure: Plotly图表对象
        """
        fig = px.bar(df, x=x, y=y, title=title)
        return fig

class SentimentVisualizer(Visualizer):
    """情感分析可视化器"""
    
    def __init__(self):
        super().__init__()
        self.color_map = {
            '正面': '#2ecc71',
            '中性': '#95a5a6',
            '负面': '#e74c3c'
        }
    
    def create_sentiment_distribution(self, sentiment_results: List[Dict]) -> go.Figure:
        """
        创建情感分布饼图
        
        Args:
            sentiment_results: 情感分析结果列表
            
        Returns:
            go.Figure: Plotly图表对象
        """
        df = pd.DataFrame(sentiment_results)
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='评论情感分布',
            color=sentiment_counts.index,
            color_discrete_map=self.color_map
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        
        return fig
    
    def create_sentiment_trend(self, df: pd.DataFrame) -> go.Figure:
        """
        创建情感趋势图
        
        Args:
            df: 包含时间戳和情感的DataFrame
            
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            # 确保timestamp列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 按日期统计各情感的数量
            daily_sentiment = df.groupby([
                df['timestamp'].dt.date,
                'sentiment'
            ]).size().unstack(fill_value=0)
            
            # 计算百分比
            daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
            
            fig = go.Figure()
            
            for sentiment in daily_sentiment_pct.columns:
                fig.add_trace(go.Scatter(
                    x=daily_sentiment_pct.index,
                    y=daily_sentiment_pct[sentiment],
                    name=sentiment,
                    mode='lines',
                    line=dict(color=self.color_map.get(sentiment)),
                    stackgroup='one'
                ))
            
            fig.update_layout(
                title='情感趋势变化',
                xaxis_title='日期',
                yaxis_title='比例 (%)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"情感趋势图生成失败：{str(e)}")
            return go.Figure()
    
    def create_rating_sentiment_comparison(self, df: pd.DataFrame) -> go.Figure:
        """
        创建评分-情感对比图
        
        Args:
            df: 包含评分和情感的DataFrame
            
        Returns:
            go.Figure: Plotly图表对象
        """
        # 计算每个评分的情感分布
        rating_sentiment = pd.crosstab(
            df['rating'],
            df['sentiment'],
            normalize='index'
        ) * 100
        
        fig = go.Figure()
        
        for sentiment in rating_sentiment.columns:
            fig.add_trace(go.Bar(
                x=rating_sentiment.index,
                y=rating_sentiment[sentiment],
                name=sentiment,
                marker_color=self.color_map.get(sentiment)
            ))
        
        fig.update_layout(
            title='评分与情感分布对比',
            xaxis_title='评分',
            yaxis_title='比例 (%)',
            barmode='stack',
            showlegend=True
        )
        
        return fig

# class KeywordVisualizer(Visualizer):
#     """关键词可视化器"""
    
#     def __init__(self):
#         """初始化关键词可视化器"""
#         super().__init__()
#         self.color_scheme = {
#             'positive': '#2ecc71',
#             'negative': '#e74c3c',
#             'neutral': '#95a5a6'
#         }
    
#     def create_wordcloud(self, keywords: Dict[str, float], 
#                         title: str = "关键词云图") -> go.Figure:
#         """
#         生成词云图
        
#         Args:
#             keywords: 关键词及其权重
#             title: 图表标题
            
#         Returns:
#             go.Figure: Plotly图表对象
#         """
#         try:
#             # 创建词云对象
#             wc = WordCloud(
#                 width=800,
#                 height=400,
#                 background_color='white',
#                 font_path='simhei.ttf'  # 使用系统中文字体
#             )
            
#             # 生成词云
#             wc.generate_from_frequencies(keywords)
            
#             # 转换为图像
#             img = wc.to_image()
            
#             # 将图像转换为base64字符串
#             img_buffer = io.BytesIO()
#             img.save(img_buffer, format='PNG')
#             img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
#             # 创建Plotly图表
#             fig = go.Figure()
            
#             fig.add_layout_image(
#                 dict(
#                     source=f'data:image/png;base64,{img_str}',
#                     x=0,
#                     y=1,
#                     sizex=1,
#                     sizey=1,
#                     sizing="stretch",
#                     layer="below"
#                 )
#             )
            
#             fig.update_layout(
#                 title=title,
#                 showlegend=False,
#                 width=800,
#                 height=400,
#                 margin=dict(l=0, r=0, t=30, b=0)
#             )
            
#             return fig
            
#         except Exception as e:
#             st.error(f"词云图生成失败：{str(e)}")
#             return go.Figure()
    
#     def create_keyword_trend_chart(self, trend_df: pd.DataFrame) -> go.Figure:
#         """
#         生成关键词趋势图
        
#         Args:
#             trend_df: 包含时间戳和关键词频率的DataFrame
            
#         Returns:
#             go.Figure: Plotly图表对象
#         """
#         try:
#             fig = go.Figure()
            
#             # 为每个关键词添加一条线
#             for keyword in trend_df['keyword'].unique():
#                 keyword_data = trend_df[trend_df['keyword'] == keyword]
                
#                 fig.add_trace(go.Scatter(
#                     x=keyword_data['timestamp'],
#                     y=keyword_data['frequency'],
#                     name=keyword,
#                     mode='lines+markers'
#                 ))
            
#             fig.update_layout(
#                 title='关键词趋势变化',
#                 xaxis_title='时间',
#                 yaxis_title='频率',
#                 hovermode='x unified',
#                 showlegend=True
#             )
            
#             return fig
            
#         except Exception as e:
#             st.error(f"趋势图生成失败：{str(e)}")
#             return go.Figure()
    
#     def create_rating_keyword_comparison(self, keywords_by_rating: Dict[str, Dict[str, float]]) -> go.Figure:
#         """
#         生成评分关键词对比图
        
#         Args:
#             keywords_by_rating: 各评分段的关键词及权重
            
#         Returns:
#             go.Figure: Plotly图表对象
#         """
#         try:
#             fig = go.Figure()
            
#             # 添加正面评价关键词
#             fig.add_trace(go.Bar(
#                 x=list(keywords_by_rating['positive'].keys()),
#                 y=list(keywords_by_rating['positive'].values()),
#                 name='正面评价',
#                 marker_color=self.color_scheme['positive']
#             ))
            
#             # 添加负面评价关键词
#             fig.add_trace(go.Bar(
#                 x=list(keywords_by_rating['negative'].keys()),
#                 y=[-v for v in keywords_by_rating['negative'].values()],
#                 name='负面评价',
#                 marker_color=self.color_scheme['negative']
#             ))
            
#             fig.update_layout(
#                 title='正负面评价关键词对比',
#                 barmode='overlay',
#                 yaxis_title='权重',
#                 showlegend=True
#             )
            
#             return fig
            
#         except Exception as e:
#             st.error(f"对比图生成失败：{str(e)}")
#             return go.Figure()


class KeywordVisualizer(Visualizer):
    def create_wordcloud(self, keywords: Dict[str, float],
                        title: str = "关键词云图") -> go.Figure:
        """创建词云图"""
        try:
            if not keywords:
                raise ValueError("没有关键词数据")
            
            # 准备数据
            words = list(keywords.keys())
            weights = list(keywords.values())
            
            # 计算字体大小和颜色
            min_size = 15
            max_size = 50
            sizes = [min_size + (max_size - min_size) * w for w in weights]
            
            # 创建颜色映射
            colors = px.colors.sequential.Viridis
            color_indices = np.linspace(0, len(colors)-1, len(words)).astype(int)
            
            # 创建散点图
            fig = go.Figure()
            
            # 使用极坐标分布词语
            theta = np.linspace(0, 2*np.pi, len(words))
            radius = np.random.uniform(0.3, 1, len(words))
            x_pos = radius * np.cos(theta)
            y_pos = radius * np.sin(theta)
            
            # 添加文本散点
            fig.add_trace(go.Scatter(
                x=x_pos,
                y=y_pos,
                text=words,
                mode='text',
                textfont=dict(
                    size=sizes,
                    color=[colors[i] for i in color_indices]
                ),
                hoverinfo='text',
                hovertemplate='%{text}<br>权重: %{customdata:.2f}<extra></extra>',
                customdata=weights
            ))
            
            # 更新布局
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    y=0.95
                ),
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-1.2, 1.2]
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-1.2, 1.2],
                    scaleanchor='x',
                    scaleratio=1
                ),
                width=800,
                height=600,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            return fig
        
        except Exception as e:
            st.error(f"词云图生成失败: {str(e)}")
            return go.Figure()

    def create_keyword_trend_chart(self, trend_df: pd.DataFrame) -> go.Figure:
        """生成关键词趋势图"""
        try:
            if trend_df.empty:
                raise ValueError("趋势数据为空")
            
            fig = go.Figure()
            
            # 为每个关键词和类别组合创建一条线
            for category in trend_df['category'].unique():
                for keyword in trend_df['keyword'].unique():
                    mask = (trend_df['category'] == category) & (trend_df['keyword'] == keyword)
                    data = trend_df[mask]
                    
                    if not data.empty:
                        name = f"{keyword} ({category})"
                        fig.add_trace(go.Scatter(
                            x=data['timestamp'],
                            y=data['frequency'],
                            name=name,
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=6)
                        ))
            
            # 更新布局
            fig.update_layout(
                title=dict(
                    text='关键词趋势变化',
                    x=0.5,
                    y=0.95
                ),
                xaxis_title='时间',
                yaxis_title='相对频率',
                hovermode='x unified',
                showlegend=True,
                width=900,
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(t=100)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"趋势图生成失败: {str(e)}")
            return go.Figure()

    def create_rating_keyword_comparison(self, 
                                       keywords_by_rating: Dict[str, Dict[str, float]]) -> go.Figure:
        """生成评分关键词对比图"""
        try:
            if not keywords_by_rating.get('positive') and not keywords_by_rating.get('negative'):
                raise ValueError("关键词数据为空")
            
            fig = go.Figure()
            
            # 添加正面评价关键词
            pos_words = list(keywords_by_rating['positive'].keys())
            pos_weights = list(keywords_by_rating['positive'].values())
            
            fig.add_trace(go.Bar(
                x=pos_words,
                y=pos_weights,
                name='正面评价',
                marker_color='rgb(46, 204, 113)',
                text=pos_weights,
                texttemplate='%{text:.2f}',
                textposition='outside'
            ))
            
            # 添加负面评价关键词
            neg_words = list(keywords_by_rating['negative'].keys())
            neg_weights = [-w for w in keywords_by_rating['negative'].values()]
            
            fig.add_trace(go.Bar(
                x=neg_words,
                y=neg_weights,
                name='负面评价',
                marker_color='rgb(231, 76, 60)',
                text=[-w for w in neg_weights],
                texttemplate='%{text:.2f}',
                textposition='outside'
            ))
            
            # 更新布局
            fig.update_layout(
                title=dict(
                    text='正负面评价关键词对比',
                    x=0.5,
                    y=0.95
                ),
                xaxis_title='关键词',
                yaxis_title='权重',
                barmode='relative',
                showlegend=True,
                width=900,
                height=600,
                margin=dict(t=100, b=100),
                yaxis=dict(
                    tickformat='.2f'
                )
            )
            
            # 添加水平参考线
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            return fig
            
        except Exception as e:
            st.error(f"对比图生成失败: {str(e)}")
            return go.Figure()

class TopicVisualizer(Visualizer):
    """主题可视化器"""
    
    def __init__(self):
        """初始化主题可视化器"""
        super().__init__()
        self.color_scheme = px.colors.qualitative.Set3
    
    def create_topic_distribution(self, topic_results: Dict) -> go.Figure:
        """
        创建主题分布图
        
        Args:
            topic_results: 主题分析结果
            
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            # 计算每个主题的文档数量
            topic_counts = pd.Series(topic_results['document_topics']).value_counts()
            
            # 创建饼图
            fig = px.pie(
                values=topic_counts.values,
                names=[f"主题{i+1}" for i in topic_counts.index],
                title="文档主题分布",
                color_discrete_sequence=self.color_scheme
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            return fig
            
        except Exception as e:
            st.error(f"主题分布图生成失败：{str(e)}")
            return go.Figure()
    
    def create_topic_network(self, topic_results: Dict) -> go.Figure:
        """
        创建主题-关键词网络图
        
        Args:
            topic_results: 主题分析结果
            
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            # 创建网络图
            G = nx.Graph()
            
            # 添加节点和边
            for topic_idx, keywords in enumerate(topic_results['topics']):
                topic_node = f'主题{topic_idx+1}'
                G.add_node(topic_node, node_type='topic')
                
                # 添加关键词节点和边
                for keyword in keywords:
                    G.add_node(keyword, node_type='keyword')
                    G.add_edge(topic_node, keyword)
            
            # 使用spring_layout布局
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # 创建节点轨迹
            node_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers+text',
                hoverinfo='text',
                marker=dict(
                    size=[],
                    color=[],
                    colorscale='Viridis',
                    line_width=2
                ),
                textposition="top center"
            )
            
            # 创建边轨迹
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # 添加边
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            # 添加节点
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
                node_trace['text'] += (node,)
                
                # 设置节点大小和颜色
                if G.nodes[node]['node_type'] == 'topic':
                    node_trace['marker']['size'] += (30,)
                    node_trace['marker']['color'] += (1,)
                else:
                    node_trace['marker']['size'] += (20,)
                    node_trace['marker']['color'] += (0,)
            
            # 创建图形
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title='主题-关键词网络图',
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
            
            return fig
            
        except Exception as e:
            st.error(f"主题网络图生成失败：{str(e)}")
            return go.Figure()
    
    def create_topic_heatmap(self, topics_df: pd.DataFrame) -> go.Figure:
        """
        创建主题热力图
        
        Args:
            topics_df: 主题-文档分布矩阵
            
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            fig = go.Figure(data=go.Heatmap(
                z=topics_df.values,
                x=[f"主题{i+1}" for i in range(topics_df.shape[1])],
                y=topics_df.index,
                colorscale='Viridis',
                colorbar=dict(title="主题权重")
            ))
            
            fig.update_layout(
                title="文档-主题分布热力图",
                xaxis_title="主题",
                yaxis_title="文档ID",
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"主题热力图生成失败：{str(e)}")
            return go.Figure()
    
    def create_topic_trend(self, trend_df: pd.DataFrame) -> go.Figure:
        """
        创建主题趋势图
        
        Args:
            trend_df: 主题趋势数据
            
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            fig = go.Figure()
            
            # 为每个主题添加一条线
            for topic in trend_df.columns:
                fig.add_trace(go.Scatter(
                    x=trend_df.index,
                    y=trend_df[topic],
                    name=f"主题{topic+1}",
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title="主题趋势变化",
                xaxis_title="时间",
                yaxis_title="比例 (%)",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"主题趋势图生成失败：{str(e)}")
            return go.Figure()

class InsightVisualizer(Visualizer):
    """洞察可视化器"""
    
    def __init__(self):
        """初始化洞察可视化器"""
        super().__init__()
        self.color_scheme = {
            'anomaly': '#e74c3c',
            'normal': '#2ecc71',
            'correlation': '#3498db',
            'trend_up': '#27ae60',
            'trend_down': '#c0392b'
        }
    
    # def create_anomaly_scatter(self, df: pd.DataFrame) -> go.Figure:
    #     """
    #     创建异常检测散点图
        
    #     Args:
    #         df: 包含异常标记的DataFrame
            
    #     Returns:
    #         go.Figure: Plotly图表对象
    #     """
    #     try:
    #         fig = go.Figure()
            
    #         # 添加正常点
    #         normal_data = df[~df['is_anomaly']]
    #         fig.add_trace(go.Scatter(
    #             x=normal_data['rating'],
    #             y=normal_data['sentiment_score'],
    #             mode='markers',
    #             name='正常评论',
    #             marker=dict(
    #                 color=self.color_scheme['normal'],
    #                 size=8,
    #                 opacity=0.6
    #             ),
    #             text=normal_data['review_text'],
    #             hovertemplate="评分: %{x}<br>情感分: %{y}<br>评论: %{text}<extra></extra>"
    #         ))
            
    #         # 添加异常点
    #         anomaly_data = df[df['is_anomaly']]
    #         fig.add_trace(go.Scatter(
    #             x=anomaly_data['rating'],
    #             y=anomaly_data['sentiment_score'],
    #             mode='markers',
    #             name='异常评论',
    #             marker=dict(
    #                 color=self.color_scheme['anomaly'],
    #                 size=10,
    #                 symbol='x'
    #             ),
    #             text=anomaly_data.apply(
    #                 lambda x: f"{x['review_text']}<br>异常原因: {x['anomaly_reason']}", 
    #                 axis=1
    #             ),
    #             hovertemplate="评分: %{x}<br>情感分: %{y}<br>%{text}<extra></extra>"
    #         ))
            
    #         fig.update_layout(
    #             title="评分-情感分布异常检测",
    #             xaxis_title="评分",
    #             yaxis_title="情感得分",
    #             hovermode='closest',
    #             showlegend=True
    #         )
            
    #         return fig
            
    #     except Exception as e:
    #         st.error(f"异常散点图生成失败：{str(e)}")
    #         return go.Figure()


    def create_anomaly_scatter(self, df: pd.DataFrame) -> go.Figure:
        """
        创建异常检测散点图
        
        Args:
            df: 包含异常标记的DataFrame
            
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            fig = go.Figure()
            
            # 确定要使用的列
            x_col = 'rating' if 'rating' in df.columns else 'timestamp'
            y_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'text_length'
            text_col = 'content' if 'content' in df.columns else 'review_text'
            
            if 'is_anomaly' not in df.columns:
                raise ValueError("DataFrame must contain 'is_anomaly' column")
                
            # 添加正常点
            normal_data = df[~df['is_anomaly']]
            if not normal_data.empty:
                fig.add_trace(go.Scatter(
                    x=normal_data[x_col],
                    y=normal_data[y_col],
                    mode='markers',
                    name='正常评论',
                    marker=dict(
                        color=self.color_scheme['normal'],
                        size=8,
                        opacity=0.6
                    ),
                    text=normal_data[text_col],
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>评论: %{{text}}<extra></extra>"
                ))
            
            # 添加异常点
            anomaly_data = df[df['is_anomaly']]
            if not anomaly_data.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_data[x_col],
                    y=anomaly_data[y_col],
                    mode='markers',
                    name='异常评论',
                    marker=dict(
                        color=self.color_scheme['anomaly'],
                        size=10,
                        symbol='x'
                    ),
                    text=anomaly_data.apply(
                        lambda x: f"{x[text_col]}<br>异常原因: {x.get('anomaly_reason', '未知')}", 
                        axis=1
                    ),
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>%{{text}}<extra></extra>"
                ))
            
            fig.update_layout(
                title="评论特征分布异常检测",
                xaxis_title=x_col,
                yaxis_title=y_col,
                hovermode='closest',
                showlegend=True
            )
            
            return fig
        
        except Exception as e:
            st.error(f"异常散点图生成失败: {str(e)}")
            return go.Figure()
    
    def create_correlation_heatmap(self, correlation_data: Dict) -> go.Figure:
        """
        创建相关性热力图
        
        Args:
            correlation_data: 相关性分析数据
            
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            # 构建相关性矩阵
            metrics = ['rating', 'sentiment', 'review_length', 'time_interval']
            matrix = np.zeros((len(metrics), len(metrics)))
            
            for i, m1 in enumerate(metrics):
                for j, m2 in enumerate(metrics):
                    key = f"{m1}_{m2}"
                    if key in correlation_data:
                        matrix[i][j] = correlation_data[key]
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=metrics,
                y=metrics,
                colorscale='RdBu',
                zmid=0,
                text=np.round(matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="评论特征相关性分析",
                xaxis_title="特征",
                yaxis_title="特征",
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"相关性热力图生成失败：{str(e)}")
            return go.Figure()
    
    def create_issue_timeline(self, issues: List[Dict]) -> go.Figure:
        """
        创建问题时间线
        
        Args:
            issues: 问题列表
            
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            # 准备数据
            df = pd.DataFrame(issues)
            df['size'] = np.log1p(df['frequency']) * 20  # 根据频率调整气泡大小
            
            fig = go.Figure()
            
            # 添加气泡图
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['avg_rating'],
                mode='markers',
                marker=dict(
                    size=df['size'],
                    color=df['avg_rating'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="平均评分")
                ),
                text=df.apply(
                    lambda x: f"关键词: {x['keyword']}<br>" +
                             f"频率: {x['frequency']}<br>" +
                             f"平均评分: {x['avg_rating']:.1f}",
                    axis=1
                ),
                hovertemplate="%{text}<extra></extra>"
            ))
            
            fig.update_layout(
                title="新出现问题时间线",
                xaxis_title="问题ID",
                yaxis_title="平均评分",
                showlegend=False,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"问题时间线生成失败：{str(e)}")
            return go.Figure()
    
    def create_improvement_dashboard(self, suggestions: List[Dict]) -> go.Figure:
        """
        创建改进建议仪表板
        
        Args:
            suggestions: 改进建议列表
            
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            # 准备数据
            df = pd.DataFrame(suggestions)
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "问题频率分布",
                    "平均评分分布",
                    "关键词网络",
                    "改进优先级"
                ),
                specs=[
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )
            
            # 1. 问题频率饼图
            fig.add_trace(
                go.Pie(
                    labels=df['topic'],
                    values=df['frequency'],
                    textinfo='percent+label'
                ),
                row=1, col=1
            )
            
            # 2. 平均评分条形统计图
            fig.add_trace(
                go.Bar(
                    x=df['topic'],
                    y=df['avg_rating'],
                    marker_color=df['avg_rating'],
                    colorscale='RdYlGn'
                ),
                row=1, col=2
            )
            
            # 3. 优先级散点图
            priority_score = (1 - df['avg_rating']/5) * df['frequency']
            fig.add_trace(
                go.Bar(
                    x=df['topic'],
                    y=priority_score,
                    marker_color='red',
                    name='优先级得分'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="改进建议分析仪表板",
                showlegend=False,
                height=800
            )
            
            return fig
            
        except Exception as e:
            st.error(f"改进建议仪表板生成失败：{str(e)}")
            return go.Figure()