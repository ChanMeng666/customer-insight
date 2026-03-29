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
    """Base visualization class"""

    def __init__(self):
        """Initialize the base visualizer"""
        pass

    def create_line_plot(self, df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """
        Create a line plot

        Args:
            df: DataFrame
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title

        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.line(df, x=x, y=y, title=title)
        return fig

    def create_bar_plot(self, df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """
        Create a bar plot

        Args:
            df: DataFrame
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title

        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.bar(df, x=x, y=y, title=title)
        return fig

class SentimentVisualizer(Visualizer):
    """Sentiment analysis visualizer"""

    def __init__(self):
        super().__init__()
        self.color_map = {
            'positive': '#2ecc71',
            'neutral': '#95a5a6',
            'negative': '#e74c3c'
        }

    def create_sentiment_distribution(self, sentiment_results: List[Dict]) -> go.Figure:
        """
        Create a sentiment distribution pie chart

        Args:
            sentiment_results: List of sentiment analysis results

        Returns:
            go.Figure: Plotly figure object
        """
        df = pd.DataFrame(sentiment_results)
        sentiment_counts = df['sentiment'].value_counts()

        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map=self.color_map
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)

        return fig

    def create_sentiment_trend(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a sentiment trend chart

        Args:
            df: DataFrame containing timestamps and sentiments

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Ensure timestamp column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Count sentiments by date
            daily_sentiment = df.groupby([
                df['timestamp'].dt.date,
                'sentiment'
            ]).size().unstack(fill_value=0)

            # Calculate percentages
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
                title='Sentiment Trend',
                xaxis_title='Date',
                yaxis_title='Proportion (%)',
                hovermode='x unified'
            )

            return fig

        except Exception as e:
            st.error(f"Failed to generate sentiment trend chart: {str(e)}")
            return go.Figure()

    def create_rating_sentiment_comparison(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a rating-sentiment comparison chart

        Args:
            df: DataFrame containing ratings and sentiments

        Returns:
            go.Figure: Plotly figure object
        """
        # Calculate sentiment distribution for each rating
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
            title='Rating vs Sentiment Distribution',
            xaxis_title='Rating',
            yaxis_title='Proportion (%)',
            barmode='stack',
            showlegend=True
        )

        return fig


class KeywordVisualizer(Visualizer):
    def create_wordcloud(self, keywords: Dict[str, float],
                        title: str = "Keyword Cloud") -> go.Figure:
        """Create a word cloud chart"""
        try:
            if not keywords:
                raise ValueError("No keyword data available")

            # Prepare data
            words = list(keywords.keys())
            weights = list(keywords.values())

            # Calculate font sizes and colors
            min_size = 15
            max_size = 50
            sizes = [min_size + (max_size - min_size) * w for w in weights]

            # Create color mapping
            colors = px.colors.sequential.Viridis
            color_indices = np.linspace(0, len(colors)-1, len(words)).astype(int)

            # Create scatter plot
            fig = go.Figure()

            # Distribute words using polar coordinates
            theta = np.linspace(0, 2*np.pi, len(words))
            radius = np.random.uniform(0.3, 1, len(words))
            x_pos = radius * np.cos(theta)
            y_pos = radius * np.sin(theta)

            # Add text scatter
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
                hovertemplate='%{text}<br>Weight: %{customdata:.2f}<extra></extra>',
                customdata=weights
            ))

            # Update layout
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
            st.error(f"Failed to generate word cloud: {str(e)}")
            return go.Figure()

    def create_keyword_trend_chart(self, trend_df: pd.DataFrame) -> go.Figure:
        """Create a keyword trend chart"""
        try:
            if trend_df.empty:
                raise ValueError("Trend data is empty")

            fig = go.Figure()

            # Create a line for each keyword and category combination
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

            # Update layout
            fig.update_layout(
                title=dict(
                    text='Keyword Trends',
                    x=0.5,
                    y=0.95
                ),
                xaxis_title='Time',
                yaxis_title='Relative Frequency',
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
            st.error(f"Failed to generate trend chart: {str(e)}")
            return go.Figure()

    def create_rating_keyword_comparison(self,
                                       keywords_by_rating: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create a rating keyword comparison chart"""
        try:
            if not keywords_by_rating.get('positive') and not keywords_by_rating.get('negative'):
                raise ValueError("Keyword data is empty")

            fig = go.Figure()

            # Add positive review keywords
            pos_words = list(keywords_by_rating['positive'].keys())
            pos_weights = list(keywords_by_rating['positive'].values())

            fig.add_trace(go.Bar(
                x=pos_words,
                y=pos_weights,
                name='Positive Reviews',
                marker_color='rgb(46, 204, 113)',
                text=pos_weights,
                texttemplate='%{text:.2f}',
                textposition='outside'
            ))

            # Add negative review keywords
            neg_words = list(keywords_by_rating['negative'].keys())
            neg_weights = [-w for w in keywords_by_rating['negative'].values()]

            fig.add_trace(go.Bar(
                x=neg_words,
                y=neg_weights,
                name='Negative Reviews',
                marker_color='rgb(231, 76, 60)',
                text=[-w for w in neg_weights],
                texttemplate='%{text:.2f}',
                textposition='outside'
            ))

            # Update layout
            fig.update_layout(
                title=dict(
                    text='Positive vs Negative Keyword Comparison',
                    x=0.5,
                    y=0.95
                ),
                xaxis_title='Keywords',
                yaxis_title='Weight',
                barmode='relative',
                showlegend=True,
                width=900,
                height=600,
                margin=dict(t=100, b=100),
                yaxis=dict(
                    tickformat='.2f'
                )
            )

            # Add horizontal reference line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            return fig

        except Exception as e:
            st.error(f"Failed to generate comparison chart: {str(e)}")
            return go.Figure()

class TopicVisualizer(Visualizer):
    """Topic visualizer"""

    def __init__(self):
        """Initialize the topic visualizer"""
        super().__init__()
        self.color_scheme = px.colors.qualitative.Set3

    def create_topic_distribution(self, topic_results: Dict) -> go.Figure:
        """
        Create a topic distribution chart

        Args:
            topic_results: Topic analysis results

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Count documents per topic
            topic_counts = pd.Series(topic_results['document_topics']).value_counts()

            # Create pie chart
            fig = px.pie(
                values=topic_counts.values,
                names=[f"Topic {i+1}" for i in topic_counts.index],
                title="Document Topic Distribution",
                color_discrete_sequence=self.color_scheme
            )

            fig.update_traces(textposition='inside', textinfo='percent+label')

            return fig

        except Exception as e:
            st.error(f"Failed to generate topic distribution chart: {str(e)}")
            return go.Figure()

    def create_topic_network(self, topic_results: Dict) -> go.Figure:
        """
        Create a topic-keyword network graph

        Args:
            topic_results: Topic analysis results

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Create network graph
            G = nx.Graph()

            # Add nodes and edges
            for topic_idx, keywords in enumerate(topic_results['topics']):
                topic_node = f'Topic {topic_idx+1}'
                G.add_node(topic_node, node_type='topic')

                # Add keyword nodes and edges
                for keyword in keywords:
                    G.add_node(keyword, node_type='keyword')
                    G.add_edge(topic_node, keyword)

            # Use spring_layout for positioning
            pos = nx.spring_layout(G, k=1, iterations=50)

            # Create node trace
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

            # Create edge trace
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            # Add edges
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)

            # Add nodes
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
                node_trace['text'] += (node,)

                # Set node size and color
                if G.nodes[node]['node_type'] == 'topic':
                    node_trace['marker']['size'] += (30,)
                    node_trace['marker']['color'] += (1,)
                else:
                    node_trace['marker']['size'] += (20,)
                    node_trace['marker']['color'] += (0,)

            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title='Topic-Keyword Network',
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))

            return fig

        except Exception as e:
            st.error(f"Failed to generate topic network graph: {str(e)}")
            return go.Figure()

    def create_topic_heatmap(self, topics_df: pd.DataFrame) -> go.Figure:
        """
        Create a topic heatmap

        Args:
            topics_df: Topic-document distribution matrix

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            fig = go.Figure(data=go.Heatmap(
                z=topics_df.values,
                x=[f"Topic {i+1}" for i in range(topics_df.shape[1])],
                y=topics_df.index,
                colorscale='Viridis',
                colorbar=dict(title="Topic Weight")
            ))

            fig.update_layout(
                title="Document-Topic Distribution Heatmap",
                xaxis_title="Topic",
                yaxis_title="Document ID",
                height=600
            )

            return fig

        except Exception as e:
            st.error(f"Failed to generate topic heatmap: {str(e)}")
            return go.Figure()

    def create_topic_trend(self, trend_df: pd.DataFrame) -> go.Figure:
        """
        Create a topic trend chart

        Args:
            trend_df: Topic trend data

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            fig = go.Figure()

            # Add a line for each topic
            for topic in trend_df.columns:
                fig.add_trace(go.Scatter(
                    x=trend_df.index,
                    y=trend_df[topic],
                    name=f"Topic {topic+1}",
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))

            fig.update_layout(
                title="Topic Trends",
                xaxis_title="Time",
                yaxis_title="Proportion (%)",
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
            st.error(f"Failed to generate topic trend chart: {str(e)}")
            return go.Figure()

class InsightVisualizer(Visualizer):
    """Insight visualizer"""

    def __init__(self):
        """Initialize the insight visualizer"""
        super().__init__()
        self.color_scheme = {
            'anomaly': '#e74c3c',
            'normal': '#2ecc71',
            'correlation': '#3498db',
            'trend_up': '#27ae60',
            'trend_down': '#c0392b'
        }


    def create_anomaly_scatter(self, df: pd.DataFrame) -> go.Figure:
        """
        Create an anomaly detection scatter plot

        Args:
            df: DataFrame containing anomaly markers

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            fig = go.Figure()

            # Determine which columns to use
            x_col = 'rating' if 'rating' in df.columns else 'timestamp'
            y_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'text_length'
            text_col = 'content' if 'content' in df.columns else 'review_text'

            if 'is_anomaly' not in df.columns:
                raise ValueError("DataFrame must contain 'is_anomaly' column")

            # Add normal points
            normal_data = df[~df['is_anomaly']]
            if not normal_data.empty:
                fig.add_trace(go.Scatter(
                    x=normal_data[x_col],
                    y=normal_data[y_col],
                    mode='markers',
                    name='Normal Reviews',
                    marker=dict(
                        color=self.color_scheme['normal'],
                        size=8,
                        opacity=0.6
                    ),
                    text=normal_data[text_col],
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>Review: %{{text}}<extra></extra>"
                ))

            # Add anomaly points
            anomaly_data = df[df['is_anomaly']]
            if not anomaly_data.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_data[x_col],
                    y=anomaly_data[y_col],
                    mode='markers',
                    name='Anomalous Reviews',
                    marker=dict(
                        color=self.color_scheme['anomaly'],
                        size=10,
                        symbol='x'
                    ),
                    text=anomaly_data.apply(
                        lambda x: f"{x[text_col]}<br>Anomaly reason: {x.get('anomaly_reason', 'Unknown')}",
                        axis=1
                    ),
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>%{{text}}<extra></extra>"
                ))

            fig.update_layout(
                title="Review Feature Anomaly Detection",
                xaxis_title=x_col,
                yaxis_title=y_col,
                hovermode='closest',
                showlegend=True
            )

            return fig

        except Exception as e:
            st.error(f"Failed to generate anomaly scatter plot: {str(e)}")
            return go.Figure()

    def create_correlation_heatmap(self, correlation_data: Dict) -> go.Figure:
        """
        Create a correlation heatmap

        Args:
            correlation_data: Correlation analysis data

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Build correlation matrix
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
                title="Review Feature Correlation Analysis",
                xaxis_title="Feature",
                yaxis_title="Feature",
                height=500
            )

            return fig

        except Exception as e:
            st.error(f"Failed to generate correlation heatmap: {str(e)}")
            return go.Figure()

    def create_issue_timeline(self, issues: List[Dict]) -> go.Figure:
        """
        Create an issue timeline

        Args:
            issues: List of issues

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Prepare data
            df = pd.DataFrame(issues)
            df['size'] = np.log1p(df['frequency']) * 20  # Adjust bubble size based on frequency

            fig = go.Figure()

            # Add bubble chart
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['avg_rating'],
                mode='markers',
                marker=dict(
                    size=df['size'],
                    color=df['avg_rating'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Average Rating")
                ),
                text=df.apply(
                    lambda x: f"Keywords: {x['keyword']}<br>" +
                             f"Frequency: {x['frequency']}<br>" +
                             f"Average Rating: {x['avg_rating']:.1f}",
                    axis=1
                ),
                hovertemplate="%{text}<extra></extra>"
            ))

            fig.update_layout(
                title="Emerging Issues Timeline",
                xaxis_title="Issue ID",
                yaxis_title="Average Rating",
                showlegend=False,
                height=400
            )

            return fig

        except Exception as e:
            st.error(f"Failed to generate issue timeline: {str(e)}")
            return go.Figure()

    def create_improvement_dashboard(self, suggestions: List[Dict]) -> go.Figure:
        """
        Create an improvement suggestions dashboard

        Args:
            suggestions: List of improvement suggestions

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Prepare data
            df = pd.DataFrame(suggestions)

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Issue Frequency Distribution",
                    "Average Rating Distribution",
                    "Keyword Network",
                    "Improvement Priority"
                ),
                specs=[
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )

            # 1. Issue frequency pie chart
            fig.add_trace(
                go.Pie(
                    labels=df['topic'],
                    values=df['frequency'],
                    textinfo='percent+label'
                ),
                row=1, col=1
            )

            # 2. Average rating bar chart
            fig.add_trace(
                go.Bar(
                    x=df['topic'],
                    y=df['avg_rating'],
                    marker_color=df['avg_rating'],
                    colorscale='RdYlGn'
                ),
                row=1, col=2
            )

            # 3. Priority scatter plot
            priority_score = (1 - df['avg_rating']/5) * df['frequency']
            fig.add_trace(
                go.Bar(
                    x=df['topic'],
                    y=priority_score,
                    marker_color='red',
                    name='Priority Score'
                ),
                row=2, col=2
            )

            fig.update_layout(
                title="Improvement Suggestions Dashboard",
                showlegend=False,
                height=800
            )

            return fig

        except Exception as e:
            st.error(f"Failed to generate improvement suggestions dashboard: {str(e)}")
            return go.Figure()
