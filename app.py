import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from src.data_processor import DataProcessor
from src.text_analyzer import SentimentAnalyzer, KeywordAnalyzer, TopicAnalyzer, InsightAnalyzer
from src.visualizer import Visualizer, SentimentVisualizer, KeywordVisualizer, TopicVisualizer, InsightVisualizer

from src.utils.jieba_config import initialize_jieba
import plotly.graph_objects as go

def display_statistics(stats: dict):
    """Display statistics metrics"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Reviews", stats['total_reviews'])
    with col2:
        st.metric("Average Rating", stats['average_rating'])
    with col3:
        st.metric("Reviews in Last 30 Days",
                 sum(count for date, count in stats['daily_reviews'].items()
                     if datetime.strptime(str(date), '%Y-%m-%d').date() >
                     datetime.now().date() - timedelta(days=30)))

def plot_rating_distribution(stats: dict):
    """Plot rating distribution chart"""
    df = pd.DataFrame(list(stats['rating_distribution'].items()),
                     columns=['Rating', 'Count'])
    fig = px.bar(df, x='Rating', y='Count',
                 title='Rating Distribution',
                 labels={'Rating': 'Rating', 'Count': 'Review Count'})
    st.plotly_chart(fig)

def plot_daily_reviews(stats: dict):
    """Plot daily review count trend chart"""
    df = pd.DataFrame(list(stats['daily_reviews'].items()),
                     columns=['Date', 'Count'])
    fig = px.line(df, x='Date', y='Count',
                  title='Review Count Trend',
                  labels={'Date': 'Date', 'Count': 'Review Count'})
    st.plotly_chart(fig)

def show_topic_analysis(df: pd.DataFrame, language: str):
    """
    Display topic analysis page

    Args:
        df: DataFrame
        language: text language
    """
    st.header("Topic Clustering Analysis")

    # Initialize analyzers
    topic_analyzer = TopicAnalyzer(language)
    topic_visualizer = TopicVisualizer()

    # Configuration area
    col1, col2, col3 = st.columns(3)
    with col1:
        n_topics = st.slider("Number of Topics", 3, 10, 5)
    with col2:
        method = st.selectbox(
            "Analysis Method",
            options=[("LDA Topic Model", "lda"), ("KMeans Clustering", "kmeans")],
            format_func=lambda x: x[0],
            index=0
        )[1]
    with col3:
        time_window = st.selectbox(
            "Time Window",
            options=[("Day", "D"), ("Week", "W"), ("Month", "M")],
            format_func=lambda x: x[0],
            index=1
        )[1]

    # Analysis results display
    tab1, tab2, tab3, tab4 = st.tabs([
        "Topic Distribution",
        "Topic Network",
        "Topic Heatmap",
        "Topic Trends"
    ])

    if st.button("Start Analysis", key="start_topic_analysis"):
        with st.spinner("Running topic analysis..."):
            # Get text data
            texts = df['content'].tolist()

            # Execute topic analysis
            topic_results = topic_analyzer.analyze_topics(
                texts,
                n_topics=n_topics,
                method=method
            )

            if topic_results:
                # Topic distribution
                with tab1:
                    st.subheader("Topic Distribution")
                    st.plotly_chart(
                        topic_visualizer.create_topic_distribution(topic_results),
                        use_container_width=True
                    )

                    # Display topic keywords
                    st.subheader("Topic Keywords")
                    for i, keywords in enumerate(topic_results['topics']):
                        with st.expander(f"Topic {i+1}"):
                            st.write("Keywords: " + ", ".join(keywords))
                            if 'example_docs' in topic_results:
                                st.write("Example documents:")
                                for doc in topic_results['example_docs'].get(i, []):
                                    st.markdown(f"> {doc}")

                # Topic network
                with tab2:
                    st.subheader("Topic-Keyword Network")
                    st.plotly_chart(
                        topic_visualizer.create_topic_network(topic_results),
                        use_container_width=True
                    )

                # Topic heatmap
                with tab3:
                    st.subheader("Topic Distribution Heatmap")
                    if 'topic_distribution' in topic_results:
                        topic_dist_df = pd.DataFrame(
                            topic_results['topic_distribution'],
                            columns=[f"Topic {i+1}" for i in range(n_topics)]
                        )
                        st.plotly_chart(
                            topic_visualizer.create_topic_heatmap(topic_dist_df),
                            use_container_width=True
                        )
                    else:
                        st.info("Current analysis method does not support topic distribution heatmap")

                # Topic trends
                with tab4:
                    st.subheader("Topic Trend Analysis")
                    trend_df = topic_analyzer.get_topic_trends(
                        df,
                        topic_results['document_topics'],
                        time_window
                    )

                    if not trend_df.empty:
                        st.plotly_chart(
                            topic_visualizer.create_topic_trend(trend_df),
                            use_container_width=True
                        )
                    else:
                        st.warning("Unable to generate topic trend chart, possibly due to insufficient data")

                # Download analysis results
                if st.download_button(
                    "Download Analysis Results",
                    data=pd.DataFrame({
                        'text': texts,
                        'topic': topic_results['document_topics']
                    }).to_csv(index=False),
                    file_name="topic_analysis_results.csv",
                    mime="text/csv"
                ):
                    st.success("Analysis results downloaded")

def show_insights_analysis(df: pd.DataFrame, language: str):
    """
    Display review insight analysis page

    Args:
        df: DataFrame
        language: text language
    """
    st.header("Review Deep Insights")

    # Initialize analyzers
    insight_analyzer = InsightAnalyzer(language)
    insight_visualizer = InsightVisualizer()

    if st.button("Start Analysis", key="start_insight_analysis"):
        with st.spinner("Running deep analysis..."):
            # Extract insights
            insights = insight_analyzer.extract_insights(df)

            if insights and 'anomalies' in insights:
                # Display anomaly detection results
                st.subheader("Anomalous Review Detection")

                # Display anomaly statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Anomalous Reviews",
                        insights['anomalies']['total']
                    )
                with col2:
                    st.metric(
                        "Anomaly Ratio",
                        f"{insights['anomalies']['total']/len(df):.1%}"
                    )

                # Display anomaly scatter plot - using full DataFrame with anomaly markers
                if 'df' in insights['anomalies']:
                    st.plotly_chart(
                        insight_visualizer.create_anomaly_scatter(insights['anomalies']['df']),
                        use_container_width=True
                    )

                # Display correlation analysis
                st.subheader("Correlation Analysis")
                if insights.get('correlations'):
                    st.metric(
                        "Rating-Sentiment Correlation",
                        f"{insights['correlations']['correlation']:.2f}"
                    )
                    st.metric(
                        "Rating-Sentiment Consistency",
                        f"{insights['correlations']['consistency']:.1%}"
                    )



def create_custom_visualizations(df: pd.DataFrame):
    """Create custom visualization charts"""
    # Chart type selection
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Time Series", "Rating Distribution", "Text Length Distribution", "Custom Group Statistics"]
    )

    # Display different options and charts based on selected chart type
    if chart_type == "Time Series":
        # Time granularity selection
        time_unit = st.selectbox(
            "Select Time Granularity",
            ["Day", "Week", "Month"],
            key="time_series_unit"
        )

        # Select metrics to display
        metrics = st.multiselect(
            "Select Metrics",
            ["Review Count", "Average Rating", "Average Text Length"],
            default=["Review Count"]
        )

        # Create time series chart
        try:
            # Set time index
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            if time_unit == "Week":
                df['date'] = pd.to_datetime(df['timestamp']).dt.to_period('W').astype(str)
            elif time_unit == "Month":
                df['date'] = pd.to_datetime(df['timestamp']).dt.to_period('M').astype(str)

            # Prepare data
            fig = go.Figure()

            if "Review Count" in metrics:
                counts = df.groupby('date').size()
                fig.add_trace(go.Scatter(
                    x=counts.index,
                    y=counts.values,
                    name="Review Count",
                    mode='lines+markers'
                ))

            if "Average Rating" in metrics:
                avg_ratings = df.groupby('date')['rating'].mean()
                fig.add_trace(go.Scatter(
                    x=avg_ratings.index,
                    y=avg_ratings.values,
                    name="Average Rating",
                    mode='lines+markers',
                    yaxis="y2"
                ))

            if "Average Text Length" in metrics:
                avg_lengths = df.groupby('date')['text_length'].mean()
                fig.add_trace(go.Scatter(
                    x=avg_lengths.index,
                    y=avg_lengths.values,
                    name="Average Text Length",
                    mode='lines+markers',
                    yaxis="y3"
                ))

            # Update layout
            fig.update_layout(
                title="Time Series Trend",
                xaxis_title="Date",
                yaxis_title="Review Count",
                yaxis2=dict(
                    title="Average Rating",
                    overlaying="y",
                    side="right"
                ),
                yaxis3=dict(
                    title="Average Text Length",
                    overlaying="y",
                    side="right",
                    position=0.95
                )
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error generating time series chart: {str(e)}")

    elif chart_type == "Rating Distribution":
        # Chart subtype selection
        subtype = st.selectbox(
            "Select Chart Subtype",
            ["Bar Chart", "Pie Chart", "Box Plot"],
            key="rating_dist_type"
        )

        try:
            if subtype == "Bar Chart":
                rating_counts = df['rating'].value_counts().sort_index()
                fig = px.bar(
                    x=rating_counts.index,
                    y=rating_counts.values,
                    title="Rating Distribution",
                    labels={'x': 'Rating', 'y': 'Count'}
                )

            elif subtype == "Pie Chart":
                rating_counts = df['rating'].value_counts()
                fig = px.pie(
                    values=rating_counts.values,
                    names=rating_counts.index,
                    title="Rating Proportion"
                )

            elif subtype == "Box Plot":
                fig = px.box(
                    df,
                    y="rating",
                    title="Rating Distribution Box Plot"
                )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error generating rating distribution chart: {str(e)}")

    elif chart_type == "Text Length Distribution":
        # Select number of bins
        num_bins = st.slider("Number of Bins", 10, 50, 20)

        try:
            fig = px.histogram(
                df,
                x="text_length",
                nbins=num_bins,
                title="Text Length Distribution",
                labels={'text_length': 'Text Length', 'count': 'Count'}
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error generating text length distribution chart: {str(e)}")

    elif chart_type == "Custom Group Statistics":
        # Select group field
        group_col = st.selectbox(
            "Select Group Field",
            ["category", "rating", "sentiment"],
            key="custom_group_col"
        )

        # Select metric
        agg_metric = st.selectbox(
            "Select Metric",
            ["Count", "Average Text Length", "Average Sentiment Score"],
            key="custom_agg_metric"
        )

        try:
            if agg_metric == "Count":
                counts = df[group_col].value_counts()
                fig = px.bar(
                    x=counts.index,
                    y=counts.values,
                    title=f"Review Count by {group_col}",
                    labels={'x': group_col, 'y': 'Count'}
                )

            elif agg_metric == "Average Text Length":
                avg_lengths = df.groupby(group_col)['text_length'].mean()
                fig = px.bar(
                    x=avg_lengths.index,
                    y=avg_lengths.values,
                    title=f"Average Text Length by {group_col}",
                    labels={'x': group_col, 'y': 'Average Length'}
                )

            elif agg_metric == "Average Sentiment Score":
                avg_sentiment = df.groupby(group_col)['sentiment_score'].mean()
                fig = px.bar(
                    x=avg_sentiment.index,
                    y=avg_sentiment.values,
                    title=f"Average Sentiment Score by {group_col}",
                    labels={'x': group_col, 'y': 'Average Sentiment Score'}
                )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error generating custom group statistics chart: {str(e)}")


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Customer Review Analysis",
        page_icon="📊",
        layout="wide"
    )

        # Initialize jieba
    try:
        initialize_jieba()
    except Exception as e:
        st.error(f"Failed to initialize jieba: {str(e)}")
        return

    # Page title
    st.title("Customer Review Analysis")
    st.markdown("### AI-powered analysis for your customer review data")

    # Initialize processor
    data_processor = DataProcessor()

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        language = st.selectbox(
            "Select review language",
            ["Chinese", "English", "Bilingual"]
        )
        language_map = {"Chinese": "chinese", "English": "english", "Bilingual": "chinese"}
        language = language_map[language]

        analysis_options = st.multiselect(
            "Select analysis dimensions",
            options=["Sentiment Analysis", "Keyword Analysis", "Topic Clustering", "Rating Statistics"],
            default=["Sentiment Analysis", "Keyword Analysis"]
        )

    # Main content area
    tabs = st.tabs(["Data Upload", "Sentiment Analysis", "Keyword Analysis", "Topic Analysis", "Insight Analysis", "Visualization"])
    
    with tabs[0]:
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx"]
        )

        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file)

                # Validate required columns exist
                required_columns = ['timestamp', 'content']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    st.error(f"CSV file is missing required columns: {', '.join(missing_columns)}")
                    st.info("Please ensure the CSV file contains these columns: timestamp and content")
                    return

                # Convert timestamp to datetime format
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    st.error(f"Failed to convert timestamp format: {str(e)}")
                    st.info("Please ensure the timestamp column is in a standard datetime format")
                    return

                # Initialize filtered_df
                filtered_df = df.copy()

                # Display data preview
                st.subheader("Data Preview")
                st.write(df.head())

                # Data filtering section
                st.subheader("Data Filtering")

                # Create three-column layout
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Date range filter
                    st.write("Date Range Filter")
                    min_date = df['timestamp'].min().date()
                    max_date = df['timestamp'].max().date()

                    start_date = st.date_input(
                        "Start Date",
                        min_value=min_date,
                        max_value=max_date,
                        value=min_date
                    )

                    end_date = st.date_input(
                        "End Date",
                        min_value=min_date,
                        max_value=max_date,
                        value=max_date
                    )

                with col2:
                    # Rating filter (if rating column exists)
                    if 'rating' in df.columns:
                        st.write("Rating Filter")
                        min_rating = float(df['rating'].min())
                        max_rating = float(df['rating'].max())

                        rating_range = st.slider(
                            "Select rating range",
                            min_value=min_rating,
                            max_value=max_rating,
                            value=(min_rating, max_rating),
                            step=0.5
                        )

                with col3:
                    # Text length filter
                    st.write("Text Length Filter")
                    df['text_length'] = df['content'].str.len()
                    min_length = int(df['text_length'].min())
                    max_length = int(df['text_length'].max())

                    length_range = st.slider(
                        "Select text length range",
                        min_value=min_length,
                        max_value=max_length,
                        value=(min_length, max_length)
                    )

                # Keyword search
                search_term = st.text_input("Search keywords (multiple keywords separated by spaces)")

                # Apply filter conditions
                filtered_df = df.copy()

                # Time filter
                filtered_df = filtered_df[
                    (filtered_df['timestamp'].dt.date >= start_date) &
                    (filtered_df['timestamp'].dt.date <= end_date)
                ]

                # Rating filter
                if 'rating' in df.columns:
                    filtered_df = filtered_df[
                        (filtered_df['rating'] >= rating_range[0]) &
                        (filtered_df['rating'] <= rating_range[1])
                    ]

                # Text length filter
                filtered_df = filtered_df[
                    (filtered_df['text_length'] >= length_range[0]) &
                    (filtered_df['text_length'] <= length_range[1])
                ]

                # Keyword search
                if search_term:
                    keywords = search_term.split()
                    search_mask = filtered_df['content'].str.contains('|'.join(keywords), case=False, na=False)
                    filtered_df = filtered_df[search_mask]

                # Display filtered data statistics
                st.subheader("Filter Results")
                col1, col2 = st.columns(2)

                with col1:
                    ratio = len(filtered_df)/len(df)
                    st.metric(
                        "Filtered Records",
                        f"{len(filtered_df)} records",
                        f"{ratio:.1%}"
                    )

                with col2:
                    if 'rating' in filtered_df.columns:
                        st.metric(
                            "Average Rating",
                            f"{filtered_df['rating'].mean():.1f}",
                            f"Original avg {df['rating'].mean():.1f}"
                        )

                # Display filtered data preview
                st.subheader("Filtered Data Preview")
                st.write(filtered_df.head())

                # Add download filtered data functionality
                if st.download_button(
                    "Download Filtered Data",
                    data=filtered_df.to_csv(index=False),
                    file_name="filtered_data.csv",
                    mime="text/csv"
                ):
                    st.success("Data downloaded successfully!")

            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
                st.info("Please ensure the CSV file is correctly formatted and contains the required columns")
    
    # Initialize analyzers
    sentiment_analyzer = SentimentAnalyzer(language)
    sentiment_visualizer = SentimentVisualizer()

    with tabs[1]:
        st.header("Sentiment Analysis")
        if 'df' not in locals():
            st.info("Please upload a data file first")
        else:
            try:
                # Initialize analyzers
                sentiment_analyzer = SentimentAnalyzer(language)
                sentiment_visualizer = SentimentVisualizer()

                # Sentiment analysis settings
                st.subheader("Analysis Settings")
                batch_size = st.slider("Batch Size", 16, 64, 32)

                if st.button("Start Analysis"):
                    # Display progress bar
                    progress_text = st.empty()
                    progress_bar = st.progress(0)

                    try:
                        # Execute sentiment analysis
                        texts = filtered_df['content'].tolist()
                        sentiment_results = SentimentAnalyzer.cached_analyze_batch(
                            texts=texts,
                            model_name=sentiment_analyzer.model_name,
                            device=str(sentiment_analyzer.device),
                            language=language,
                            batch_size=batch_size
                        )

                        # Update DataFrame
                        filtered_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
                        filtered_df['confidence'] = [r['confidence'] for r in sentiment_results]

                        # Calculate statistics
                        sentiment_stats = sentiment_analyzer.get_sentiment_stats(sentiment_results)

                        # Display results
                        st.subheader("Analysis Results")

                        # Display basic statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Positive Review Ratio",
                                f"{sentiment_stats['sentiment_distribution'].get('positive', 0) / len(sentiment_results):.1%}"
                            )
                        with col2:
                            st.metric(
                                "Negative Review Ratio",
                                f"{sentiment_stats['sentiment_distribution'].get('negative', 0) / len(sentiment_results):.1%}"
                            )
                        with col3:
                            st.metric(
                                "Average Confidence",
                                f"{sentiment_stats['average_confidence']:.2f}"
                            )

                        # Display visualization results
                        st.plotly_chart(
                            sentiment_visualizer.create_sentiment_distribution(sentiment_results)
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(
                                sentiment_visualizer.create_sentiment_trend(filtered_df)
                            )
                        with col2:
                            st.plotly_chart(
                                sentiment_visualizer.create_rating_sentiment_comparison(filtered_df)
                            )

                        # Display typical reviews
                        st.subheader("Typical Review Examples")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("##### Typical Positive Reviews")
                            for comment in sentiment_stats['typical_positive']:
                                st.markdown(f"""
                                > {comment['text']}
                                > Confidence: {comment['confidence']:.2f}
                                """)

                        with col2:
                            st.markdown("##### Typical Negative Reviews")
                            for comment in sentiment_stats['typical_negative']:
                                st.markdown(f"""
                                > {comment['text']}
                                > Confidence: {comment['confidence']:.2f}
                                """)

                    except Exception as e:
                        st.error(f"Error during sentiment analysis: {str(e)}")
            except Exception as e:
                st.error(f"Error initializing sentiment analyzer: {str(e)}")
    
    with tabs[2]:
        if 'df' not in locals():
            st.info("Please upload a data file first")
        else:
            try:
                show_keyword_analysis(filtered_df, language)
            except Exception as e:
                st.error(f"Error during keyword analysis: {str(e)}")

    with tabs[3]:
        if 'df' not in locals():
            st.info("Please upload a data file first")
        else:
            try:
                show_topic_analysis(filtered_df, language)
            except Exception as e:
                st.error(f"Error during topic analysis: {str(e)}")

    with tabs[4]:
        if 'df' not in locals():
            st.info("Please upload a data file first")
        else:
            try:
                show_insights_analysis(filtered_df, language)
            except Exception as e:
                st.error(f"Error during insight analysis: {str(e)}")

    with tabs[5]:
        st.header("Visualization")
        if 'df' not in locals():
            st.info("Please upload a data file first")
        else:
            try:
                st.subheader("Custom Charts")
                create_custom_visualizations(filtered_df)
            except Exception as e:
                st.error(f"Error during visualization: {str(e)}")

def show_keyword_analysis(df: pd.DataFrame, language: str, suffix: str = "default"):
    """
    Display keyword analysis page (with unique keys)

    Args:
        df: DataFrame
        language: text language
        suffix: suffix for generating unique keys
    """
    st.header("Keyword Analysis")

    # Initialize analyzers
    keyword_analyzer = KeywordAnalyzer(language)
    keyword_visualizer = KeywordVisualizer()

    # Analysis settings
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider(
            "Number of keywords",
            5, 50, 20,
            key=f"keyword_analysis_count_slider_{suffix}"
        )
    with col2:
        time_window = st.selectbox(
            "Time Window",
            options=[("Day", "D"), ("Week", "W"), ("Month", "M")],
            format_func=lambda x: x[0],
            index=1,
            key=f"keyword_analysis_time_window_{suffix}"
        )[1]

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Word Cloud", "Keyword Trends", "Rating Keyword Comparison"])

    with tab1:
        st.subheader("Word Cloud")
        if st.button("Generate Word Cloud", key=f"keyword_analysis_wordcloud_button_{suffix}"):
            with st.spinner("Generating word cloud..."):
                texts = df['content'].tolist()
                keywords = keyword_analyzer.extract_keywords(texts, top_n)
                st.plotly_chart(
                    keyword_visualizer.create_wordcloud(keywords),
                    use_container_width=True
                )

    with tab2:
        st.subheader("Keyword Trend Analysis")
        if st.button("Analyze Trends", key=f"keyword_analysis_trend_button_{suffix}"):
            with st.spinner("Analyzing trends..."):
                all_keywords = keyword_analyzer.extract_keywords(
                    df['content'].tolist(),
                    top_n=10
                )
                trend_df = keyword_analyzer.calculate_keyword_trends(
                    df,
                    list(all_keywords.keys()),
                    time_window
                )
                st.plotly_chart(
                    keyword_visualizer.create_keyword_trend_chart(trend_df),
                    use_container_width=True
                )

    with tab3:
        st.subheader("Rating Keyword Comparison")
        if st.button("Analyze Rating Keywords", key=f"keyword_analysis_rating_button_{suffix}"):
            with st.spinner("Analyzing rating keywords..."):
                keywords_by_rating = keyword_analyzer.extract_keywords_by_rating(
                    df,
                    top_n=10
                )
                st.plotly_chart(
                    keyword_visualizer.create_rating_keyword_comparison(keywords_by_rating),
                    use_container_width=True
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Top-Rated Keywords")
                    for word, weight in keywords_by_rating['positive'].items():
                        st.write(f"- {word}: {weight:.4f}")

                with col2:
                    st.markdown("##### Low-Rated Keywords")
                    for word, weight in keywords_by_rating['negative'].items():
                        st.write(f"- {word}: {weight:.4f}")

if __name__ == "__main__":
    main() 