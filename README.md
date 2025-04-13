# Age Group Detection from Text

This project uses Natural Language Processing (NLP) and Machine Learning (ML) to detect the age group of a person based on their written text (comments, blog posts, etc.). The system analyzes linguistic patterns, vocabulary choices, and writing styles to predict which age category the author likely belongs to.

## Project Overview

The Age Group Detection system analyzes text input and classifies it into different age categories:
- Under 18
- 18-24 years
- 25-34 years
- 35-49 years
- 50+ years

This can be useful for:
- Content personalization
- Marketing research
- Social media analysis
- Demographic studies
- User experience customization

## Features

- Text preprocessing using NLP techniques (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization for feature extraction
- Multiple ML classifiers (Logistic Regression, Random Forest, SVM)
- Interactive Streamlit web application
- Visualization of prediction probabilities
- Example texts for each age group

## Technical Stack

- **Python**: Core programming language
- **NLP Libraries**: NLTK, spaCy
- **Machine Learning**: scikit-learn, TensorFlow
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Application**: Streamlit

## Project Structure

```
nlp_age_detection/
├── data/               # Dataset files
│   ├── raw/            # Original dataset
│   └── processed/      # Preprocessed data
├── models/             # Trained model files
├── src/                # Source code
│   ├── generate_dataset.py     # Data generation script
│   ├── explore_preprocess.py   # Data exploration and preprocessing
│   └── train_model.py          # Model training and evaluation
├── app/                # Streamlit application
│   └── app.py          # Main application file
├── README.md           # Project documentation
└── requirements.txt    # Dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nlp-age-detection.git
   cd nlp-age-detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app/app.py
   ```

## Usage

1. Enter or paste text in the input field
2. Click "Analyze" to process the text
3. View the predicted age group and confidence scores
4. Explore visualizations of prediction probabilities
5. Try example texts from different age groups

## Model Development

The model was developed through the following steps:

1. **Data Collection**: Created a synthetic dataset with 5,000 text samples across 5 age groups
2. **Preprocessing**: Applied text cleaning, tokenization, stopword removal, and lemmatization
3. **Feature Extraction**: Used TF-IDF vectorization to convert text to numerical features
4. **Model Training**: Trained multiple classifiers (Logistic Regression, Random Forest, SVM)
5. **Hyperparameter Tuning**: Optimized model parameters using grid search
6. **Evaluation**: Achieved high accuracy in age group classification

## Results

The model achieved excellent performance on the test dataset:
- Logistic Regression: 100% accuracy
- Random Forest: 99.9% accuracy
- SVM: 100% accuracy

The high accuracy is expected given the synthetic nature of the dataset with clear linguistic patterns for each age group. In real-world applications with more nuanced text, we would expect lower but still useful accuracy.

## Future Improvements

- Expand the dataset with real-world text samples
- Implement more sophisticated NLP techniques (word embeddings, transformers)
- Add multi-language support
- Improve prediction accuracy for edge cases
- Enhance the UI with more detailed analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK and spaCy for NLP tools
- scikit-learn for machine learning algorithms
- Streamlit for the web application framework
