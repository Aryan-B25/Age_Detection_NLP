import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set file paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'age_classifier_model.joblib')

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Text preprocessing function (same as in training)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Function to predict age group
def predict_age_group(text, model):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Make prediction
    prediction = model.predict([processed_text])[0]
    probabilities = model.predict_proba([processed_text])[0]
    
    return prediction, probabilities, model.classes_

# Function to create probability chart
def create_probability_chart(probabilities, classes):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a bar chart
    sns.barplot(x=classes, y=probabilities, ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities by Age Group')
    
    # Add percentage labels on top of bars
    for i, p in enumerate(probabilities):
        ax.text(i, p + 0.01, f'{p:.2%}', ha='center')
    
    # Set y-axis limit to make room for percentage labels
    ax.set_ylim(0, max(probabilities) + 0.1)
    
    plt.tight_layout()
    return fig

# Age group descriptions
age_group_descriptions = {
    '<18': "This text likely comes from a teenager or younger person. The language typically includes modern slang, informal expressions, and topics related to school, social media trends, and youth culture.",
    '18-24': "This text appears to be from a young adult, possibly a college student or recent graduate. Common themes include education, early career concerns, apartment hunting, dating, and establishing independence.",
    '25-34': "This text suggests a young professional in their late twenties or early thirties. Topics often include career development, relationships, possibly starting a family, financial planning, and work-life balance.",
    '35-49': "This text indicates a middle-aged adult. Common themes include established career, family responsibilities, homeownership, children's education, and long-term planning.",
    '50+': "This text likely comes from an older adult. Topics often include retirement planning or retirement life, grandchildren, health concerns, reflections on the past, and changing technology."
}

# Example texts for each age group
example_texts = {
    '<18': "omg this is so lit! no cap, that new tiktok trend is fire. lowkey obsessed with it. that's sus tho.",
    '18-24': "just finished my finals and now apartment hunting is so stressful. dating apps are the worst but my student loans are killing me.",
    '25-34': "trying to balance work and personal life. mortgage rates are insane right now! thinking about changing careers but need to schedule doctor appointments.",
    '35-49': "my daughter is graduating college next month. planning for retirement while managing aging parents and grown kids is challenging.",
    '50+': "retirement is wonderful! grandchildren are such a blessing. remember when phones had cords? technology moves too fast these days."
}

# Streamlit app
def main():
    st.set_page_config(page_title="Age Group Detector", page_icon="👤", layout="wide")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # App title and description
    st.title("Age Group Detector")
    st.markdown("""
    This application uses Natural Language Processing (NLP) and Machine Learning to predict the age group of a person based on their writing style.
    Enter some text in the box below to see which age group the author likely belongs to.
    """)
    
    # Sidebar with information
    st.sidebar.title("About")
    st.sidebar.info("""
    This app analyzes text to predict the author's age group based on language patterns, vocabulary, and topics.
    
    **Age Groups:**
    - Under 18
    - 18-24 years
    - 25-34 years
    - 35-49 years
    - 50+ years
    
    The model was trained on a dataset of text samples with known age groups.
    """)
    
    # Example selector in sidebar
    st.sidebar.title("Try an Example")
    selected_example = st.sidebar.selectbox(
        "Select an age group to see an example:",
        list(example_texts.keys())
    )
    
    if st.sidebar.button("Load Example"):
        st.session_state.text_input = example_texts[selected_example]
    
    # Initialize session state for text input if it doesn't exist
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # Text input
    text_input = st.text_area("Enter text to analyze:", value=st.session_state.text_input, height=150)
    
    # Update session state when text changes
    st.session_state.text_input = text_input
    
    # Analyze button
    if st.button("Analyze Text"):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                try:
                    # Make prediction
                    prediction, probabilities, classes = predict_age_group(text_input, model)
                    
                    # Display results
                    st.success(f"Predicted Age Group: **{prediction}**")
                    
                    # Display description of the age group
                    st.markdown("### Analysis")
                    st.markdown(age_group_descriptions[prediction])
                    
                    # Create two columns for the results
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Display probability chart
                        st.markdown("### Prediction Probabilities")
                        fig = create_probability_chart(probabilities, classes)
                        st.pyplot(fig)
                    
                    with col2:
                        # Display probability table
                        st.markdown("### Detailed Probabilities")
                        prob_df = pd.DataFrame({
                            'Age Group': classes,
                            'Probability': probabilities
                        })
                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                        st.table(prob_df)
                    
                    # Text preprocessing details
                    with st.expander("See text preprocessing details"):
                        st.markdown("### Original Text")
                        st.write(text_input)
                        
                        st.markdown("### Preprocessed Text")
                        st.write(preprocess_text(text_input))
                        
                        st.markdown("""
                        **Preprocessing steps:**
                        1. Convert to lowercase
                        2. Remove punctuation
                        3. Tokenize into words
                        4. Remove common stopwords
                        5. Lemmatize words to their base form
                        """)
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
