# Sentiment Analysis with Twitter Tweets

This repository contains two Jupyter notebooks that demonstrate how to perform sentiment analysis on Twitter tweets using Python. The notebooks cover data preprocessing, model training, and evaluation using a Logistic Regression model.

## Files

1. **Sentiment_Analysis_w_Twitter_tweets_workshop_file .ipynb**
   - This notebook includes the following steps:
     - Importing necessary libraries (Pandas, NumPy, NLTK, Scikit-learn).
     - Loading the dataset (`twitter_sentiment_data.csv`).
     - Preprocessing the text data (lowercasing, removing non-alphabet characters, and stopwords).
     - Converting text data into numerical features using `CountVectorizer`.
     - Splitting the data into training and testing sets.
     - Training a Logistic Regression model.
     - Evaluating the model's performance using accuracy and a classification report.
     - Testing the model with a custom tweet.

2. **Sentiment_Analysis_w_Twitter_tweets_workshop_file.ipynb**
   - This notebook continues from the first one and includes additional steps or variations in the analysis.

## Requirements

To run the notebooks, you need the following Python libraries:

- pandas
- numpy
- nltk
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy nltk scikit-learn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Open the Jupyter notebooks in your preferred environment (e.g., Jupyter Lab, Jupyter Notebook, or Google Colab).

3. Run the cells in the notebooks sequentially to perform sentiment analysis on the Twitter dataset.

## Dataset

The dataset used in this project is `twitter_sentiment_data.csv`, which contains Twitter messages and their corresponding sentiment labels. The sentiment labels are as follows:

- `-1`: Negative
- `0`: Neutral
- `1`: Positive
- `2`: Extremely Positive

## Results

The Logistic Regression model achieved an accuracy of approximately 53% on the test set. The classification report provides detailed metrics for each sentiment class.

## Custom Tweet Testing

You can test the model with custom tweets by modifying the `custom_tweet` variable in the notebook. The model will predict the sentiment of the custom tweet and output whether it is positive or negative.
