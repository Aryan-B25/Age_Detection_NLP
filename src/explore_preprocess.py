import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Load the dataset
df = pd.read_csv('/home/ubuntu/nlp_age_detection/data/raw/full_dataset.csv')

# Basic dataset information
print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Age group distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', data=df, palette='viridis')
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.savefig('/home/ubuntu/nlp_age_detection/data/age_group_distribution.png')
plt.close()

# Comment length analysis
df['comment_length'] = df['comment'].apply(len)
plt.figure(figsize=(10, 6))
sns.boxplot(x='age_group', y='comment_length', data=df, palette='viridis')
plt.title('Comment Length by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Comment Length (characters)')
plt.savefig('/home/ubuntu/nlp_age_detection/data/comment_length_by_age.png')
plt.close()

# Word count analysis
df['word_count'] = df['comment'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.boxplot(x='age_group', y='word_count', data=df, palette='viridis')
plt.title('Word Count by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Word Count')
plt.savefig('/home/ubuntu/nlp_age_detection/data/word_count_by_age.png')
plt.close()

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Text preprocessing function
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
    
    return tokens

# Apply preprocessing to comments
df['processed_tokens'] = df['comment'].apply(preprocess_text)
df['processed_text'] = df['processed_tokens'].apply(lambda x: ' '.join(x))

# Save preprocessed data
df.to_csv('/home/ubuntu/nlp_age_detection/data/processed/preprocessed_data.csv', index=False)

# Most common words by age group
plt.figure(figsize=(15, 20))
for i, age_group in enumerate(df['age_group'].unique()):
    plt.subplot(5, 1, i+1)
    
    # Get all words from this age group
    words = ' '.join(df[df['age_group'] == age_group]['processed_text']).split()
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Get the 15 most common words
    common_words = pd.DataFrame(word_freq.most_common(15), columns=['word', 'count'])
    
    # Plot
    sns.barplot(x='count', y='word', data=common_words, palette='viridis')
    plt.title(f'Most Common Words in Age Group: {age_group}')
    plt.tight_layout()

plt.savefig('/home/ubuntu/nlp_age_detection/data/common_words_by_age.png')
plt.close()

print("\nPreprocessing complete. Preprocessed data saved to '/home/ubuntu/nlp_age_detection/data/processed/preprocessed_data.csv'")
print("Visualization images saved to the data directory.")
