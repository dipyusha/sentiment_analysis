# Twitter Sentiment Analysis Pipeline

A complete natural language processing (NLP) and machine learning pipeline built to extract, clean, and analyze public sentiment from Twitter data. I developed this project to analyze public perception regarding COVID-19 vaccines (using a Pfizer dataset from Kaggle), cleanly classifying text into Positive, Negative, or Neutral sentiment categories.

---

## 📌 Project Overview
* **Objective**: Build an end-to-end Python pipeline to process raw text data, programmatically generate sentiment labels, extract features, and train optimized machine learning models.
* **The Challenge**: Raw social media text is noisy and lacks predefined sentiment labels. I resolved this by integrating a rule-based lexicon (`TextBlob`) to calculate text polarity and programmatically generate target categories before training my classifiers.

---

## 🛠️ Architecture & Core Workflow

### 1. Technical Stack
* **Data Manipulation**: Pandas, NumPy
* **NLP & Text Processing**: Regular Expressions (`re`), NLTK (`word_tokenize`, `PorterStemmer`, English stop words)
* **Sentiment Analysis**: TextBlob
* **Machine Learning & Tuning**: Scikit-Learn (Logistic Regression, Support Vector Machines, GridSearchCV)
* **Data Visualization**: Matplotlib, Seaborn, WordCloud

### 2. Implementation Steps

1. **Exploratory Data Analysis**: I loaded the dataset, checked for structural integrity using `df.info()`, isolated null entries, and filtered out non-textual data columns to extract a clean working matrix of 11,020 tweets.
2. **Custom Text Preprocessing Pipeline**: I wrote a modular cleaning function that reduced data noise and dropped duplicates (bringing the dataset from 11,020 entries to 10,543 unique samples) by executing:
   * Case normalization (converting all text to lowercase).
   * Removing URLs, hashtags, and special characters via custom Regular Expression (`re`) patterns.
   * Filtering out common English stop words.
   * Implementing the **Porter Stemmer** to map words down to their base grammatical roots.
3. **Lexicon-Based Sentiment Labeling**: Because the data was unlabelled, I fed the cleaned text through `TextBlob` to extract continuous polarity scores, binning them into clear targets: Positive, Negative, and Neutral.
4. **Data Visualization**: I designed Seaborn count plots, exploded pie charts, and categorical `WordClouds` to analyze and verify word frequency distributions across each sentiment class.
5. **Feature Vectorization**: I vectorized the text blocks into a numerical **Bigram Language Model** using Scikit-Learn’s `CountVectorizer`.
6. **Model Training & Hyperparameter Optimization**: I split the dataset into an 80% training and 20% testing distribution to benchmark multiple models:
   * **Logistic Regression**: Built a baseline classifier which achieved an initial **84.6% accuracy**.
   * **Tuned Logistic Regression**: Used `GridSearchCV` to optimize the inverse regularization parameter (`C`).
   * **Support Vector Machine (SVM)**: Trained a Support Vector Classifier (SVC) to test non-linear decision boundaries.
   * **Tuned SVM**: Applied `GridSearchCV` on the SVM parameters to achieve my final, highest-performing model.
7. **Evaluation**: I used `ConfusionMatrixDisplay` and standard classification reports to evaluate true positives, true negatives, and misclassification rates across all models.

---

## 📊 Core Findings & Results
* Cleaning the data and removing duplicate tweets removed roughly 500 noisy entries, ensuring the models trained on high-quality text vectors.
* Comparing the final evaluation metrics showed that the **hyperparameter-tuned Support Vector Machine (SVM)** yielded the highest predictive accuracy overall.
