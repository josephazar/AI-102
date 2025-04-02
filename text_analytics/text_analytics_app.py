import streamlit as st
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import altair as alt
import requests
import json
from io import StringIO
import re
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# Load environment variables
load_dotenv()

# Get Azure configuration
COG_SERVICE_ENDPOINT = os.getenv('COG_SERVICE_ENDPOINT')
COG_SERVICE_KEY = os.getenv('COG_SERVICE_KEY')

# Initialize Azure client
def get_azure_client():
    if not COG_SERVICE_ENDPOINT or not COG_SERVICE_KEY:
        st.error("Azure credentials not found! Please ensure your .env file contains COG_SERVICE_ENDPOINT and COG_SERVICE_KEY values.")
        return None
    
    try:
        credentials = AzureKeyCredential(COG_SERVICE_KEY)
        client = TextAnalyticsClient(endpoint=COG_SERVICE_ENDPOINT, credential=credentials)
        return client
    except Exception as e:
        st.error(f"Error initializing Azure client: {str(e)}")
        return None

# Sample data functionality
def load_sample_data(data_type="reviews"):
    if data_type == "reviews":
        reviews = [
            {"id": "1", "text": "I absolutely loved this product! It exceeded all my expectations and the customer service was exceptional. Would definitely recommend to friends and family."},
            {"id": "2", "text": "The product is okay but not worth the price. Delivery was prompt but packaging could be improved. Might consider buying again if there's a discount."},
            {"id": "3", "text": "Terrible experience with this product. It broke within a week and customer service was unresponsive. Save your money and look elsewhere."},
            {"id": "4", "text": "Apple's new iPhone 14 has an impressive camera and the battery life is much better than previous models. The ProMotion display is also a game changer."},
            {"id": "5", "text": "The Microsoft Surface laptop is sleek and powerful, perfect for both work and creative projects. Windows 11 runs smoothly on it."},
            {"id": "6", "text": "Our stay at the Grand Hotel in Seattle was magnificent. The staff was friendly and the rooms had a beautiful view of Mount Rainier. We visited the Space Needle and Pike Place Market."},
            {"id": "7", "text": "Este producto es excelente. La calidad es superior y el precio es razonable. Lo recomendaría a cualquier persona interesada en tecnología."},
            {"id": "8", "text": "Ich habe eine wunderbare Zeit in Berlin verbracht. Das Wetter war perfekt und die Sehenswürdigkeiten waren beeindruckend."},
            {"id": "9", "text": "La Tour Eiffel est magnifique la nuit. Paris est vraiment la ville de l'amour et de la lumière."},
            {"id": "10", "text": "このレストランの寿司は東京で最高です。サービスも素晴らしく、価格も手頃です。"},
        ]
        return reviews
    elif data_type == "news":
        news_articles = [
            {"id": "1", "text": "SpaceX successfully launched its latest rocket yesterday, marking a significant milestone in commercial space travel. Elon Musk announced plans for a Mars mission by 2026."},
            {"id": "2", "text": "The Federal Reserve raised interest rates by 0.25% today, citing concerns about inflation. Wall Street responded with mixed results as the Dow Jones fell slightly while NASDAQ showed gains."},
            {"id": "3", "text": "Scientists at MIT have developed a new artificial intelligence system that can predict protein structures with unprecedented accuracy, potentially revolutionizing drug discovery and development."},
            {"id": "4", "text": "Climate change concerns continue to grow as record temperatures were recorded across Europe this summer. The EU announced new initiatives to reduce carbon emissions by 55% by 2030."},
            {"id": "5", "text": "The World Health Organization declared the end of the latest Ebola outbreak in Central Africa. Medical teams will remain in the region to monitor for any new cases and support local healthcare systems."},
        ]
        return news_articles
    elif data_type == "social_media":
        posts = [
            {"id": "1", "text": "Just finished reading 'The Alchemist' by Paulo Coelho and it completely changed my perspective on following your dreams! #bookrecommendation #inspiration"},
            {"id": "2", "text": "Can't believe how bad the customer service was at @BigRetailer today. Waited for 45 minutes and the staff was so rude! #disappointed #badservice"},
            {"id": "3", "text": "The new update to this app is terrible! So many bugs and the new interface is confusing. Please fix this ASAP! @AppDeveloper"},
            {"id": "4", "text": "Just witnessed the most beautiful sunset at Malibu Beach. Nature is truly amazing! #blessed #naturelovers #california"},
            {"id": "5", "text": "Excited to announce that I'll be starting my new position at Microsoft next month! #newjob #career #microsoft"},
            {"id": "6", "text": "The latest episode of Stranger Things was mind-blowing! Can't wait for the season finale next week! No spoilers please. #StrangerThings #Netflix"},
            {"id": "7", "text": "Trying out this new vegan restaurant in downtown Seattle. The food is incredible and the atmosphere is so cozy! #vegan #foodie #Seattle"},
        ]
        return posts
    return []

# Text Analytics Features
def process_in_batches(client, documents, operation_func, batch_size=5):
    """
    Process documents in batches to comply with Azure Text Analytics service limits.
    
    Args:
        client: The Text Analytics client
        documents: List of documents to process
        operation_func: Function to call on each batch (e.g., client.analyze_sentiment)
        batch_size: Maximum number of documents per batch
        
    Returns:
        List of combined results
    """
    if not client or not documents:
        return None
    
    all_results = []
    
    try:
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = operation_func(documents=batch)
            all_results.extend(batch_results)
        
        return all_results
    except Exception as e:
        st.error(f"Error processing batch: {str(e)}")
        return None

def analyze_sentiment(client, documents):
    if not client:
        return None
    
    try:
        return process_in_batches(client, documents, client.analyze_sentiment)
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return None

def extract_key_phrases(client, documents):
    if not client:
        return None
    
    try:
        return process_in_batches(client, documents, client.extract_key_phrases)
    except Exception as e:
        st.error(f"Error extracting key phrases: {str(e)}")
        return None

def detect_language(client, documents):
    if not client:
        return None
    
    try:
        return process_in_batches(client, documents, client.detect_language)
    except Exception as e:
        st.error(f"Error detecting language: {str(e)}")
        return None

def recognize_entities(client, documents):
    if not client:
        return None
    
    try:
        return process_in_batches(client, documents, client.recognize_entities)
    except Exception as e:
        st.error(f"Error recognizing entities: {str(e)}")
        return None

def recognize_linked_entities(client, documents):
    if not client:
        return None
    
    try:
        return process_in_batches(client, documents, client.recognize_linked_entities)
    except Exception as e:
        st.error(f"Error recognizing linked entities: {str(e)}")
        return None

# Visualization functions
def plot_sentiment_distribution(sentiment_results):
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
    confidence_scores = {"positive": [], "neutral": [], "negative": []}
    
    for document in sentiment_results:
        if document.is_error:
            continue
        
        sentiment_counts[document.sentiment] += 1
        
        # Collect confidence scores
        confidence_scores["positive"].append(document.confidence_scores.positive)
        confidence_scores["neutral"].append(document.confidence_scores.neutral)
        confidence_scores["negative"].append(document.confidence_scores.negative)
    
    # Create sentiment distribution chart
    fig1 = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Sentiment Distribution",
        color=list(sentiment_counts.keys()),
        color_discrete_map={
            "positive": "#4CAF50",
            "neutral": "#2196F3",
            "negative": "#F44336",
            "mixed": "#FF9800"
        }
    )
    
    # Create confidence scores chart
    avg_scores = {
        "sentiment": ["Positive", "Neutral", "Negative"],
        "score": [
            sum(confidence_scores["positive"]) / len(confidence_scores["positive"]) if confidence_scores["positive"] else 0,
            sum(confidence_scores["neutral"]) / len(confidence_scores["neutral"]) if confidence_scores["neutral"] else 0,
            sum(confidence_scores["negative"]) / len(confidence_scores["negative"]) if confidence_scores["negative"] else 0
        ]
    }
    
    fig2 = px.bar(
        avg_scores,
        x="sentiment",
        y="score",
        title="Average Confidence Scores",
        color="sentiment",
        color_discrete_map={
            "Positive": "#4CAF50",
            "Neutral": "#2196F3",
            "Negative": "#F44336"
        }
    )
    
    return fig1, fig2

def create_entities_chart(entities_results):
    entity_categories = {}
    
    for document in entities_results:
        if document.is_error:
            continue
            
        for entity in document.entities:
            if entity.category in entity_categories:
                entity_categories[entity.category] += 1
            else:
                entity_categories[entity.category] = 1
    
    if not entity_categories:
        return None
        
    fig = px.bar(
        x=list(entity_categories.keys()),
        y=list(entity_categories.values()),
        title="Entity Categories",
        labels={"x": "Category", "y": "Count"}
    )
    
    return fig

def create_key_phrases_wordcloud(key_phrases_results):
    all_phrases = []
    
    for document in key_phrases_results:
        if document.is_error:
            continue
            
        all_phrases.extend(document.key_phrases)
    
    if not all_phrases:
        return None
        
    # Count frequency of each phrase
    phrase_counts = Counter(all_phrases)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=50
    ).generate_from_frequencies(phrase_counts)
    
    # Convert to matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    
    return fig

def create_language_chart(language_results):
    languages = {}
    
    for document in language_results:
        if document.is_error:
            continue
            
        lang_name = document.primary_language.name
        if lang_name in languages:
            languages[lang_name] += 1
        else:
            languages[lang_name] = 1
    
    if not languages:
        return None
        
    fig = px.pie(
        values=list(languages.values()),
        names=list(languages.keys()),
        title="Language Distribution"
    )
    
    return fig

# Dashboard
def social_media_monitoring_dashboard(client, posts):
    st.subheader("Social Media Monitoring Dashboard")
    
    # Process posts
    with st.spinner("Analyzing social media posts..."):
        # Prepare documents for Azure
        documents = [{"id": post["id"], "text": post["text"]} for post in posts]
        
        # Run analysis
        sentiment_results = analyze_sentiment(client, documents)
        entities_results = recognize_entities(client, documents)
        key_phrases_results = extract_key_phrases(client, documents)
        
        # Process results for dashboard
        if sentiment_results and entities_results and key_phrases_results:
            # Create DataFrame for display
            posts_df = pd.DataFrame(posts)
            
            # Add sentiment
            posts_df["sentiment"] = [doc.sentiment for doc in sentiment_results]
            posts_df["pos_score"] = [doc.confidence_scores.positive for doc in sentiment_results]
            posts_df["neu_score"] = [doc.confidence_scores.neutral for doc in sentiment_results]
            posts_df["neg_score"] = [doc.confidence_scores.negative for doc in sentiment_results]
            
            # Create a color scheme for sentiment
            sentiment_colors = {
                "positive": "#4CAF50",
                "neutral": "#2196F3",
                "negative": "#F44336",
                "mixed": "#FF9800"
            }
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pos_count = sum(1 for doc in sentiment_results if doc.sentiment == "positive")
                st.metric("Positive Posts", pos_count, f"{pos_count/len(posts)*100:.1f}%")
                
            with col2:
                neg_count = sum(1 for doc in sentiment_results if doc.sentiment == "negative")
                st.metric("Negative Posts", neg_count, f"{neg_count/len(posts)*100:.1f}%")
                
            with col3:
                # Count unique entities
                entity_count = sum(len(doc.entities) for doc in entities_results)
                st.metric("Entities Detected", entity_count)
                
            with col4:
                # Count unique key phrases
                phrase_count = sum(len(doc.key_phrases) for doc in key_phrases_results)
                st.metric("Key Phrases", phrase_count)
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, _ = plot_sentiment_distribution(sentiment_results)
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                entity_chart = create_entities_chart(entities_results)
                if entity_chart:
                    st.plotly_chart(entity_chart, use_container_width=True)
            
            # Display word cloud of key phrases
            st.subheader("Key Topics")
            wordcloud_fig = create_key_phrases_wordcloud(key_phrases_results)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            
            # Display posts with sentiment
            st.subheader("Analyzed Posts")
            for i, row in posts_df.iterrows():
                with st.container():
                    color = sentiment_colors[row["sentiment"]]
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-left: 5px solid {color}; margin-bottom: 10px; background-color: rgba(0,0,0,0.05);">
                        <p style="margin-bottom: 5px;">{row["text"]}</p>
                        <p style="font-size: 0.8em; color: #666;">
                        Sentiment: <span style="color: {color};">{row["sentiment"].title()}</span> 
                        (Positive: {row["pos_score"]:.2f}, Neutral: {row["neu_score"]:.2f}, Negative: {row["neg_score"]:.2f})
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.error("Unable to analyze posts. Please check your Azure credentials and try again.")

def customer_review_analyzer(client, reviews):
    st.subheader("Customer Review Analyzer")
    
    # Process reviews
    with st.spinner("Analyzing customer reviews..."):
        # Prepare documents for Azure
        documents = [{"id": review["id"], "text": review["text"]} for review in reviews]
        
        # Detect languages
        language_results = detect_language(client, documents)
        
        # Prepare documents with language information
        if language_results:
            documents_with_language = []
            for i, doc in enumerate(documents):
                if not language_results[i].is_error:
                    documents_with_language.append({
                        "id": doc["id"],
                        "text": doc["text"],
                        "language": language_results[i].primary_language.iso6391_name
                    })
                else:
                    documents_with_language.append({
                        "id": doc["id"],
                        "text": doc["text"],
                        "language": "en"  # Default to English if detection fails
                    })
        else:
            documents_with_language = documents
        
        # Run analysis
        sentiment_results = analyze_sentiment(client, documents_with_language)
        entities_results = recognize_entities(client, documents_with_language)
        key_phrases_results = extract_key_phrases(client, documents_with_language)
        linked_entities_results = recognize_linked_entities(client, documents_with_language)
        
        # Process results for dashboard
        if sentiment_results and entities_results and key_phrases_results:
            # Create DataFrame for display
            reviews_df = pd.DataFrame(reviews)
            
            # Add sentiment
            reviews_df["sentiment"] = [doc.sentiment for doc in sentiment_results]
            reviews_df["pos_score"] = [doc.confidence_scores.positive for doc in sentiment_results]
            reviews_df["neu_score"] = [doc.confidence_scores.neutral for doc in sentiment_results]
            reviews_df["neg_score"] = [doc.confidence_scores.negative for doc in sentiment_results]
            
            # Add language
            if language_results:
                reviews_df["language"] = [doc.primary_language.name for doc in language_results]
                reviews_df["language_score"] = [doc.primary_language.confidence_score for doc in language_results]
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pos_count = sum(1 for doc in sentiment_results if doc.sentiment == "positive")
                st.metric("Positive Reviews", pos_count, f"{pos_count/len(reviews)*100:.1f}%")
                
            with col2:
                neg_count = sum(1 for doc in sentiment_results if doc.sentiment == "negative")
                st.metric("Negative Reviews", neg_count, f"{neg_count/len(reviews)*100:.1f}%")
                
            with col3:
                # Count detected languages
                if language_results:
                    language_count = len(set(doc.primary_language.name for doc in language_results))
                    st.metric("Languages Detected", language_count)
                
            with col4:
                # Count entities
                entity_count = sum(len(doc.entities) for doc in entities_results)
                st.metric("Entities Mentioned", entity_count)
            
            # Create visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Analysis", "Key Phrases", "Entities", "Language Detection"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fig1, _ = plot_sentiment_distribution(sentiment_results)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Detailed sentiment analysis
                    sentences_sentiment = []
                    for doc in sentiment_results:
                        for sentence in doc.sentences:
                            sentences_sentiment.append({
                                "text": sentence.text[:50] + "..." if len(sentence.text) > 50 else sentence.text,
                                "sentiment": sentence.sentiment,
                                "positive": sentence.confidence_scores.positive,
                                "neutral": sentence.confidence_scores.neutral,
                                "negative": sentence.confidence_scores.negative
                            })
                    
                    if sentences_sentiment:
                        sentences_df = pd.DataFrame(sentences_sentiment)
                        fig = px.scatter(
                            sentences_df,
                            x="positive",
                            y="negative",
                            color="sentiment",
                            hover_data=["text"],
                            title="Sentence-Level Sentiment Analysis",
                            color_discrete_map={
                                "positive": "#4CAF50",
                                "neutral": "#2196F3",
                                "negative": "#F44336",
                                "mixed": "#FF9800"
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                wordcloud_fig = create_key_phrases_wordcloud(key_phrases_results)
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                
                # Show key phrases per document
                st.subheader("Key Phrases by Review")
                for i, doc in enumerate(key_phrases_results):
                    if not doc.is_error and doc.key_phrases:
                        st.markdown(f"**Review {i+1}**: {', '.join(doc.key_phrases)}")
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    entity_chart = create_entities_chart(entities_results)
                    if entity_chart:
                        st.plotly_chart(entity_chart, use_container_width=True)
                
                with col2:
                    # Create linked entities chart
                    linked_entities = {}
                    for doc in linked_entities_results:
                        if doc.is_error:
                            continue
                        
                        for entity in doc.entities:
                            if entity.name in linked_entities:
                                linked_entities[entity.name] += 1
                            else:
                                linked_entities[entity.name] = 1
                    
                    if linked_entities:
                        # Sort by count
                        sorted_entities = dict(sorted(linked_entities.items(), key=lambda x: x[1], reverse=True)[:10])
                        
                        fig = px.bar(
                            x=list(sorted_entities.keys()),
                            y=list(sorted_entities.values()),
                            title="Top Linked Entities",
                            labels={"x": "Entity", "y": "Count"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                if language_results:
                    language_chart = create_language_chart(language_results)
                    if language_chart:
                        st.plotly_chart(language_chart, use_container_width=True)
                    
                    # Display language confidence
                    lang_data = []
                    for doc in language_results:
                        if not doc.is_error:
                            lang_data.append({
                                "Document": f"Doc {doc.id}",
                                "Language": doc.primary_language.name,
                                "Confidence": doc.primary_language.confidence_score
                            })
                    
                    if lang_data:
                        lang_df = pd.DataFrame(lang_data)
                        fig = px.bar(
                            lang_df,
                            x="Document",
                            y="Confidence",
                            color="Language",
                            title="Language Detection Confidence"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Display reviews with sentiment
            st.subheader("Analyzed Reviews")
            sentiment_colors = {
                "positive": "#4CAF50",
                "neutral": "#2196F3",
                "negative": "#F44336",
                "mixed": "#FF9800"
            }
            
            for i, row in reviews_df.iterrows():
                with st.container():
                    color = sentiment_colors[row["sentiment"]]
                    
                    # Display language if available
                    language_info = f"Language: {row['language']} ({row['language_score']:.2f})" if "language" in row else ""
                    
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-left: 5px solid {color}; margin-bottom: 10px; background-color: rgba(0,0,0,0.05);">
                        <p style="margin-bottom: 5px;">{row["text"]}</p>
                        <p style="font-size: 0.8em; color: #666;">
                        Sentiment: <span style="color: {color};">{row["sentiment"].title()}</span> 
                        (Positive: {row["pos_score"]:.2f}, Neutral: {row["neu_score"]:.2f}, Negative: {row["neg_score"]:.2f})
                        {f" • {language_info}" if language_info else ""}
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.error("Unable to analyze reviews. Please check your Azure credentials and try again.")

def news_article_analyzer(client, articles):
    st.subheader("News Article Analyzer")
    
    # Process articles
    with st.spinner("Analyzing news articles..."):
        # Prepare documents for Azure
        documents = [{"id": article["id"], "text": article["text"]} for article in articles]
        
        # Run analysis
        sentiment_results = analyze_sentiment(client, documents)
        entities_results = recognize_entities(client, documents)
        key_phrases_results = extract_key_phrases(client, documents)
        linked_entities_results = recognize_linked_entities(client, documents)
        
        if sentiment_results and entities_results and key_phrases_results and linked_entities_results:
            # Create DataFrame for display
            articles_df = pd.DataFrame(articles)
            
            # Add sentiment
            articles_df["sentiment"] = [doc.sentiment for doc in sentiment_results]
            
            # Display visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, _ = plot_sentiment_distribution(sentiment_results)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Create entity network visualization
                entity_links = []
                
                for doc_id, doc in enumerate(linked_entities_results):
                    if doc.is_error:
                        continue
                    
                    for entity in doc.entities:
                        entity_links.append({
                            "Article": f"Article {doc.id}",
                            "Entity": entity.name,
                            "Type": entity.data_source,
                            "URL": entity.url,
                            "Value": 1
                        })
                
                if entity_links:
                    entity_df = pd.DataFrame(entity_links)
                    
                    # Count entities by type
                    entity_types = entity_df.groupby("Type").size().reset_index(name="Count")
                    
                    fig = px.pie(
                        entity_types,
                        values="Count",
                        names="Type",
                        title="Entity Types in News Articles"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Key phrases
            st.subheader("Key Topics in News Articles")
            wordcloud_fig = create_key_phrases_wordcloud(key_phrases_results)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            
            # Entity visualization
            st.subheader("Named Entities")
            all_entities = []
            
            for doc in entities_results:
                if doc.is_error:
                    continue
                    
                for entity in doc.entities:
                    all_entities.append({
                        "Text": entity.text,
                        "Category": entity.category,
                        "Subcategory": entity.subcategory if entity.subcategory else "N/A",
                        "Confidence": entity.confidence_score
                    })
            
            if all_entities:
                entity_df = pd.DataFrame(all_entities)
                
                # Filter to entities with high confidence
                high_conf_entities = entity_df[entity_df["Confidence"] > 0.75]
                
                if not high_conf_entities.empty:
                    fig = px.scatter(
                        high_conf_entities,
                        x="Confidence",
                        y="Category",
                        color="Category",
                        hover_data=["Text", "Subcategory"],
                        size="Confidence",
                        title="Entity Confidence by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display articles with entity highlighting
            st.subheader("Articles with Entity Analysis")
            
            for i, article in enumerate(articles):
                with st.expander(f"Article {i+1} - {articles_df.loc[i, 'sentiment'].title()} Sentiment"):
                    # Get entities for this article
                    if i < len(entities_results) and not entities_results[i].is_error:
                        article_entities = entities_results[i].entities
                        text = article["text"]
                        
                        # Create HTML with highlighted entities
                        html_text = text
                        entity_colors = {
                            "Person": "#FF9800",
                            "Location": "#4CAF50",
                            "Organization": "#2196F3",
                            "Quantity": "#9C27B0",
                            "DateTime": "#607D8B",
                            "Event": "#E91E63",
                            "Product": "#00BCD4"
                        }
                        
                        # Sort entities by position to avoid highlighting issues
                        entities_sorted = sorted(article_entities, key=lambda e: -len(e.text))
                        
                        for entity in entities_sorted:
                            color = entity_colors.get(entity.category, "#757575")
                            highlight = f'<span style="background-color: {color}40; padding: 2px; border-radius: 3px; border-bottom: 2px solid {color};" title="{entity.category}">{entity.text}</span>'
                            html_text = html_text.replace(entity.text, highlight)
                        
                        st.markdown(f'<div style="line-height: 1.6;">{html_text}</div>', unsafe_allow_html=True)
                        
                        # Show entity legend
                        st.markdown("<div style='margin-top: 10px;'><b>Entity Legend:</b></div>", unsafe_allow_html=True)
                        legend_html = ""
                        for category, color in entity_colors.items():
                            legend_html += f'<span style="margin-right: 10px;"><span style="background-color: {color}; width: 12px; height: 12px; display: inline-block; margin-right: 4px;"></span> {category}</span>'
                        
                        st.markdown(f'<div style="display: flex; flex-wrap: wrap;">{legend_html}</div>', unsafe_allow_html=True)
                    else:
                        st.write(article["text"])
                        
                    # Show key phrases
                    if i < len(key_phrases_results) and not key_phrases_results[i].is_error:
                        key_phrases = key_phrases_results[i].key_phrases
                        if key_phrases:
                            st.markdown(f"**Key Phrases:** {', '.join(key_phrases)}")
        else:
            st.error("Unable to analyze articles. Please check your Azure credentials and try again.")

def text_input_analyzer(client):
    st.subheader("Custom Text Analysis")
    
    # Text input area
    text = st.text_area("Enter text to analyze:", 
                     value="Azure Cognitive Services are cloud-based artificial intelligence services that help developers build intelligent applications without having direct AI or data science skills or knowledge. Azure Cognitive Services enable developers to easily add cognitive features into their applications.",
                     height=150)
    
    # Language selection
    language_options = {
        "Auto Detect": None,
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Japanese": "ja",
        "Chinese": "zh"
    }
    
    selected_lang = st.selectbox("Language:", list(language_options.keys()))
    language_code = language_options[selected_lang]
    
    # Analysis options
    analysis_options = st.multiselect(
        "Select analyses to perform:",
        ["Sentiment Analysis", "Key Phrase Extraction", "Entity Recognition", "Linked Entity Recognition", "Language Detection"],
        default=["Sentiment Analysis", "Entity Recognition"]
    )
    
    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing text..."):
            # Prepare document
            document = {"id": "1", "text": text}
            if language_code:
                document["language"] = language_code
            
            # Perform selected analyses
            results = {}
            
            if "Language Detection" in analysis_options or selected_lang == "Auto Detect":
                language_result = detect_language(client, [document])
                if language_result and not language_result[0].is_error:
                    detected_lang = language_result[0].primary_language
                    results["language"] = {
                        "name": detected_lang.name,
                        "iso": detected_lang.iso6391_name,
                        "confidence": detected_lang.confidence_score
                    }
                    
                    # Update document language if auto-detect was selected
                    if selected_lang == "Auto Detect":
                        document["language"] = detected_lang.iso6391_name
            
            if "Sentiment Analysis" in analysis_options:
                sentiment_result = analyze_sentiment(client, [document])
                if sentiment_result and not sentiment_result[0].is_error:
                    results["sentiment"] = {
                        "overall": sentiment_result[0].sentiment,
                        "positive": sentiment_result[0].confidence_scores.positive,
                        "neutral": sentiment_result[0].confidence_scores.neutral,
                        "negative": sentiment_result[0].confidence_scores.negative,
                        "sentences": []
                    }
                    
                    for sentence in sentiment_result[0].sentences:
                        results["sentiment"]["sentences"].append({
                            "text": sentence.text,
                            "sentiment": sentence.sentiment,
                            "positive": sentence.confidence_scores.positive,
                            "neutral": sentence.confidence_scores.neutral,
                            "negative": sentence.confidence_scores.negative
                        })
            
            if "Key Phrase Extraction" in analysis_options:
                key_phrases_result = extract_key_phrases(client, [document])
                if key_phrases_result and not key_phrases_result[0].is_error:
                    results["key_phrases"] = key_phrases_result[0].key_phrases
            
            if "Entity Recognition" in analysis_options:
                entities_result = recognize_entities(client, [document])
                if entities_result and not entities_result[0].is_error:
                    results["entities"] = []
                    for entity in entities_result[0].entities:
                        results["entities"].append({
                            "text": entity.text,
                            "category": entity.category,
                            "subcategory": entity.subcategory,
                            "confidence": entity.confidence_score
                        })
            
            if "Linked Entity Recognition" in analysis_options:
                linked_entities_result = recognize_linked_entities(client, [document])
                if linked_entities_result and not linked_entities_result[0].is_error:
                    results["linked_entities"] = []
                    for entity in linked_entities_result[0].entities:
                        results["linked_entities"].append({
                            "name": entity.name,
                            "url": entity.url,
                            "data_source": entity.data_source,
                            "matches": [{"text": m.text, "confidence": m.confidence_score} for m in entity.matches]
                        })
            
            # Display results
            if results:
                st.success("Analysis complete!")
                
                # Create tabs for different analyses
                tabs = []
                if "sentiment" in results:
                    tabs.append("Sentiment Analysis")
                if "key_phrases" in results:
                    tabs.append("Key Phrases")
                if "entities" in results or "linked_entities" in results:
                    tabs.append("Entities")
                if "language" in results:
                    tabs.append("Language")
                
                if tabs:
                    tab_views = st.tabs(tabs)
                    
                    tab_index = 0
                    if "sentiment" in results and tab_index < len(tab_views):
                        with tab_views[tab_index]:
                            sentiment = results["sentiment"]["overall"]
                            sentiment_color = {
                                "positive": "#4CAF50",
                                "neutral": "#2196F3",
                                "negative": "#F44336",
                                "mixed": "#FF9800"
                            }.get(sentiment, "#757575")
                            
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                st.markdown(f"<h3 style='color:{sentiment_color};'>{sentiment.title()}</h3>", unsafe_allow_html=True)
                            
                            with col2:
                                # Create sentiment score gauges
                                fig = go.Figure()
                                
                                fig.add_trace(go.Indicator(
                                    mode="gauge+number",
                                    value=results["sentiment"]["positive"] * 100,
                                    title={"text": "Positive"},
                                    gauge={
                                        "axis": {"range": [0, 100]},
                                        "bar": {"color": "#4CAF50"},
                                        "steps": [
                                            {"range": [0, 30], "color": "rgba(76, 175, 80, 0.2)"},
                                            {"range": [30, 70], "color": "rgba(76, 175, 80, 0.5)"},
                                            {"range": [70, 100], "color": "rgba(76, 175, 80, 0.8)"}
                                        ]
                                    },
                                    domain={"row": 0, "column": 0}
                                ))
                                
                                fig.add_trace(go.Indicator(
                                    mode="gauge+number",
                                    value=results["sentiment"]["negative"] * 100,
                                    title={"text": "Negative"},
                                    gauge={
                                        "axis": {"range": [0, 100]},
                                        "bar": {"color": "#F44336"},
                                        "steps": [
                                            {"range": [0, 30], "color": "rgba(244, 67, 54, 0.2)"},
                                            {"range": [30, 70], "color": "rgba(244, 67, 54, 0.5)"},
                                            {"range": [70, 100], "color": "rgba(244, 67, 54, 0.8)"}
                                        ]
                                    },
                                    domain={"row": 0, "column": 1}
                                ))
                                
                                fig.update_layout(
                                    grid={"rows": 1, "columns": 2, "pattern": "independent"},
                                    height=200
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col3:
                                # Create a bar chart for sentence sentiment
                                sentences = results["sentiment"]["sentences"]
                                if sentences:
                                    sentence_df = pd.DataFrame([
                                        {"Sentence": f"Sentence {i+1}", "Positive": s["positive"], "Neutral": s["neutral"], "Negative": s["negative"]}
                                        for i, s in enumerate(sentences)
                                    ])
                                    
                                    fig = px.bar(
                                        sentence_df,
                                        x="Sentence",
                                        y=["Positive", "Negative", "Neutral"],
                                        title="Sentence-Level Sentiment",
                                        barmode="group",
                                        color_discrete_map={
                                            "Positive": "#4CAF50",
                                            "Neutral": "#2196F3",
                                            "Negative": "#F44336"
                                        }
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Display sentence-level analysis
                            st.subheader("Sentence-Level Analysis")
                            for i, sentence in enumerate(results["sentiment"]["sentences"]):
                                sent_color = {
                                    "positive": "#4CAF50",
                                    "neutral": "#2196F3",
                                    "negative": "#F44336",
                                    "mixed": "#FF9800"
                                }.get(sentence["sentiment"], "#757575")
                                
                                st.markdown(
                                    f"""
                                    <div style="padding: 10px; border-left: 5px solid {sent_color}; margin-bottom: 10px; background-color: rgba(0,0,0,0.05);">
                                    <p style="margin-bottom: 5px;">{sentence["text"]}</p>
                                    <p style="font-size: 0.8em; color: #666;">
                                    Sentiment: <span style="color: {sent_color};">{sentence["sentiment"].title()}</span> 
                                    (Positive: {sentence["positive"]:.2f}, Neutral: {sentence["neutral"]:.2f}, Negative: {sentence["negative"]:.2f})
                                    </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        
                        tab_index += 1
                    
                    if "key_phrases" in results and tab_index < len(tab_views):
                        with tab_views[tab_index]:
                            if results["key_phrases"]:
                                # Create a word cloud
                                phrase_counts = Counter(results["key_phrases"])
                                
                                wordcloud = WordCloud(
                                    width=800,
                                    height=400,
                                    background_color="white",
                                    colormap="viridis",
                                    max_words=50
                                ).generate_from_frequencies({phrase: 1 for phrase in results["key_phrases"]})
                                
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis("off")
                                
                                st.pyplot(fig)
                                
                                # Display as list
                                st.markdown("### Extracted Key Phrases")
                                st.write(", ".join(results["key_phrases"]))
                            else:
                                st.info("No key phrases were extracted from the text.")
                        
                        tab_index += 1
                    
                    if ("entities" in results or "linked_entities" in results) and tab_index < len(tab_views):
                        with tab_views[tab_index]:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if "entities" in results and results["entities"]:
                                    st.markdown("### Named Entities")
                                    
                                    # Create entity table
                                    entity_df = pd.DataFrame(results["entities"])
                                    
                                    # Create bar chart of entity categories
                                    category_counts = entity_df["category"].value_counts().reset_index()
                                    category_counts.columns = ["Category", "Count"]
                                    
                                    fig = px.bar(
                                        category_counts,
                                        x="Category",
                                        y="Count",
                                        title="Entity Categories",
                                        color="Category"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display entity table
                                    st.dataframe(entity_df[["text", "category", "subcategory", "confidence"]])
                                else:
                                    st.info("No entities were recognized in the text.")
                            
                            with col2:
                                if "linked_entities" in results and results["linked_entities"]:
                                    st.markdown("### Linked Entities")
                                    
                                    for entity in results["linked_entities"]:
                                        with st.container():
                                            st.markdown(f"**{entity['name']}** ({entity['data_source']})")
                                            st.markdown(f"URL: [{entity['url']}]({entity['url']})")
                                            
                                            for match in entity["matches"]:
                                                st.markdown(f"- \"{match['text']}\" (Confidence: {match['confidence']:.2f})")
                                else:
                                    st.info("No linked entities were recognized in the text.")
                                    
                            # Display text with highlighted entities
                            if "entities" in results and results["entities"]:
                                st.markdown("### Text with Highlighted Entities")
                                
                                # Create HTML with highlighted entities
                                html_text = text
                                entity_colors = {
                                    "Person": "#FF9800",
                                    "Location": "#4CAF50",
                                    "Organization": "#2196F3",
                                    "Quantity": "#9C27B0",
                                    "DateTime": "#607D8B",
                                    "Event": "#E91E63",
                                    "Product": "#00BCD4"
                                }
                                
                                # Sort entities by length to avoid highlighting issues
                                entities_sorted = sorted(results["entities"], key=lambda e: -len(e["text"]))
                                
                                for entity in entities_sorted:
                                    color = entity_colors.get(entity["category"], "#757575")
                                    highlight = f'<span style="background-color: {color}40; padding: 2px; border-radius: 3px; border-bottom: 2px solid {color};" title="{entity["category"]}">{entity["text"]}</span>'
                                    html_text = html_text.replace(entity["text"], highlight)
                                
                                st.markdown(f'<div style="line-height: 1.6;">{html_text}</div>', unsafe_allow_html=True)
                                
                                # Show entity legend
                                st.markdown("<div style='margin-top: 10px;'><b>Entity Legend:</b></div>", unsafe_allow_html=True)
                                legend_html = ""
                                for category, color in entity_colors.items():
                                    legend_html += f'<span style="margin-right: 10px;"><span style="background-color: {color}; width: 12px; height: 12px; display: inline-block; margin-right: 4px;"></span> {category}</span>'
                                
                                st.markdown(f'<div style="display: flex; flex-wrap: wrap;">{legend_html}</div>', unsafe_allow_html=True)
                        
                        tab_index += 1
                    
                    if "language" in results and tab_index < len(tab_views):
                        with tab_views[tab_index]:
                            st.markdown(f"### Detected Language: {results['language']['name']}")
                            st.markdown(f"**ISO Code:** {results['language']['iso']}")
                            st.markdown(f"**Confidence:** {results['language']['confidence']:.4f}")
                            
                            # Create a gauge for confidence
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=results["language"]["confidence"] * 100,
                                title={"text": "Language Detection Confidence"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {"color": "#2196F3"},
                                    "steps": [
                                        {"range": [0, 50], "color": "rgba(33, 150, 243, 0.2)"},
                                        {"range": [50, 80], "color": "rgba(33, 150, 243, 0.5)"},
                                        {"range": [80, 100], "color": "rgba(33, 150, 243, 0.8)"}
                                    ]
                                }
                            ))
                            
                            fig.update_layout(height=300)
                            
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No results were returned. Try a different text or analysis options.")

# Main function for text analytics page
def show_text_analytics():
    st.title("Azure Text Analytics")
    st.write("Explore the capabilities of Azure Text Analytics to extract insights from your text data.")
    
    # Initialize Azure client
    client = get_azure_client()
    
    # Tab navigation for different use cases
    tab1, tab2, tab3, tab4 = st.tabs([
        "Custom Text Analysis", 
        "Customer Review Analyzer", 
        "Social Media Monitor",
        "News Article Analyzer"
    ])
    
    with tab1:
        if client:
            text_input_analyzer(client)
        else:
            st.error("Azure Text Analytics client could not be initialized. Please check your credentials.")
    
    with tab2:
        if client:
            # Load sample reviews
            reviews = load_sample_data("reviews")
            
            # Option to upload own data
            upload_option = st.radio("Choose data source:", ["Use sample data", "Upload your own data"])
            
            if upload_option == "Upload your own data":
                uploaded_file = st.file_uploader("Upload a CSV file with reviews (should have 'id' and 'text' columns):", type=["csv"])
                
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if "id" not in df.columns or "text" not in df.columns:
                            st.error("CSV must contain 'id' and 'text' columns.")
                        else:
                            reviews = df.to_dict('records')
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
            
            # Show the customer review analyzer
            customer_review_analyzer(client, reviews)
        else:
            st.error("Azure Text Analytics client could not be initialized. Please check your credentials.")
    
    with tab3:
        if client:
            # Load sample social media posts
            posts = load_sample_data("social_media")
            
            # Option to upload own data
            upload_option = st.radio("Choose data source:", ["Use sample data", "Upload your own data"], key="social_media_upload")
            
            if upload_option == "Upload your own data":
                uploaded_file = st.file_uploader("Upload a CSV file with social media posts (should have 'id' and 'text' columns):", type=["csv"], key="social_media_uploader")
                
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if "id" not in df.columns or "text" not in df.columns:
                            st.error("CSV must contain 'id' and 'text' columns.")
                        else:
                            posts = df.to_dict('records')
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
            
            # Show the social media monitoring dashboard
            social_media_monitoring_dashboard(client, posts)
        else:
            st.error("Azure Text Analytics client could not be initialized. Please check your credentials.")
    
    with tab4:
        if client:
            # Load sample news articles
            articles = load_sample_data("news")
            
            # Option to upload own data
            upload_option = st.radio("Choose data source:", ["Use sample data", "Upload your own data"], key="news_upload")
            
            if upload_option == "Upload your own data":
                uploaded_file = st.file_uploader("Upload a CSV file with news articles (should have 'id' and 'text' columns):", type=["csv"], key="news_uploader")
                
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if "id" not in df.columns or "text" not in df.columns:
                            st.error("CSV must contain 'id' and 'text' columns.")
                        else:
                            articles = df.to_dict('records')
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
            
            # Show the news article analyzer
            news_article_analyzer(client, articles)
        else:
            st.error("Azure Text Analytics client could not be initialized. Please check your credentials.")
    
 