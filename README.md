# SMS Spam Detection (TF-IDF + Logistic Regression)

An NLP project that classifies an SMS message as **spam** or **ham** (not spam) using TF-IDF features and a Logistic Regression model.

## Dataset
- SMS Spam Collection Dataset (Kaggle, publisher: uciml)
- 5572 messages, 2 useful columns: label (ham / spam) and message
- Class counts: ham = 4825, spam = 747 (imbalanced data)

## Project Workflow
1. Load data and keep the 2 useful columns
2. Convert labels to numbers (spam = 1, ham = 0)
3. Clean text (lowercase, remove punctuation and numbers)
4. Train / test split: 4457 train, 1115 test (stratified, so spam ratio stays the same)
5. TF-IDF vectorization (top 3000 words, English stopwords removed)
6. Train a Logistic Regression model
7. Evaluate with accuracy, classification report and confusion matrix

## Results
- Test accuracy: **0.9659 (about 0.97)**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| ham   | 0.96      | 1.00   | 0.98     | 966     |
| spam  | 0.99      | 0.75   | 0.85     | 149     |

Confusion matrix:

|              | Predicted ham | Predicted spam |
|--------------|---------------|----------------|
| Actual ham   | 965           | 1              |
| Actual spam  | 37            | 112            |

- Precision for spam is 0.99, so when the model says spam it is almost always right.
- Recall for spam is 0.75, so it misses some spam messages (37 of 149). The data is imbalanced, so accuracy alone is not enough.

## Example Prediction
Input: `Congratulations! You won a free prize, call now to claim`
Output: `1` (spam)

## How to Run
```
pip install -r requirements.txt
python spam_detection.py
```
Keep `spam.csv` in the same folder as `spam_detection.py`.

## Project Structure
```
SMS-Spam-Detection-TFIDF-LogisticRegression/
    spam.csv
    spam_detection.py
    requirements.txt
    README.md
```

## Tools
Python, Pandas, NumPy, Scikit-learn

## Author
Pragati Gaddewar
