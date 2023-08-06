import streamlit as st
from textblob import TextBlob
from transformers import pipeline
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

# Suppress the warning message
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Set the title
st.title("Sentiment Analysis App")

# List of random questions
questions = [
    "How do you feel when you spend time with your loved ones?",
    "What is a memorable achievement that makes you proud?",
    "Describe a place that brings you joy and calmness.",
    "Share an experience that made you feel grateful.",
    "What's your reaction when you encounter unexpected kindness?",
    "How do you handle challenges or setbacks in life?",
    "Talk about a moment that made you feel inspired or motivated.",
    "Describe a personal accomplishment that brought you happiness.",
    "How do you cope with stress and pressure?",
    "Reflect on a time when you overcame a difficult situation."
]

# Initialize sentiment analyzer using Hugging Face's BERT model fine-tuned on a sentiment dataset
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Initialize a session state to store sentiment scores
if 'sentiment_scores' not in st.session_state:
    st.session_state.sentiment_scores = []

# Display random questions and collect user answers
user_answers = []
for i in range(10):
    st.header(f"Question {i + 1}:")
    question = st.write(questions[i])
    
    # Get user's answer
    answer_key = f'answer_{i}'  # Unique key for each text_input
    user_answer = st.text_input("Your Answer:", key=answer_key)
    user_answers.append(user_answer)

# Calculate sentiment scores and display results
if st.button("Calculate Sentiment"):
    # Analyze sentiment using TextBlob and Hugging Face's model
    sentiment_scores = []
    for answer in user_answers:
        blob = TextBlob(answer)
        textblob_sentiment = blob.sentiment.polarity
        
        hf_sentiment = sentiment_analyzer(answer)[0]
        hf_score = hf_sentiment['score']
        
        sentiment_score = (textblob_sentiment + hf_score) / 2
        sentiment_scores.append(sentiment_score)
    
    # Store sentiment scores in session state
    st.session_state.sentiment_scores = sentiment_scores
    
    # Display sentiment analysis results
    st.header("Sentiment Analysis Results:")
    df = pd.DataFrame({
        "Question": questions,
        "Sentiment Score": sentiment_scores
    })
    st.dataframe(df)
    
    # Calculate positive and negative sentiment percentages
    positive_count = sum(1 for score in sentiment_scores if score >= 0)
    negative_count = 10 - positive_count
    positive_percentage = (positive_count / 10) * 100
    negative_percentage = (negative_count / 10) * 100
    
    # Display overall sentiment percentages
    st.header("Overall Sentiment:")
    st.write(f"Positive Sentiment: {positive_percentage:.2f}%")
    st.write(f"Negative Sentiment: {negative_percentage:.2f}%")
    
    # Display sentiment visualization
    fig, ax = plt.subplots()
    sns.barplot(x=[f"Question {i+1}" for i in range(10)], y=sentiment_scores, ax=ax)
    ax.set_title("Sentiment Analysis for Each Question")
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Question")
    st.pyplot(fig)
    
    # Display sentiment emojis
    emojis = ["😄", "😊", "😐", "😔", "😢"]
    sentiment_emojis = [emojis[int(score * 2 + 2)] for score in sentiment_scores]
    st.header("Sentiment Emojis:")
    st.write(" ".join(sentiment_emojis))
