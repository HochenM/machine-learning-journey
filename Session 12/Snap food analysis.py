import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parsivar import Tokenizer, Normalizer, FindStems

from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# ===== 1. Load Data =====
train_df = pd.read_csv('train.csv', delimiter="\t")
X = train_df['comment']
y = train_df['label_id']

# ===== 2. FIX: Initialize tools BEFORE the function =====
normalizer = Normalizer()
my_tokenizer = Tokenizer()

# If you don't have a stopwords file yet, we use an empty list so it doesn't crash.
# If you do, load it here.
stopwords = []

# ===== 3. FIX: Safely define the preprocessing function =====
def preprocess(text):
    # Added string conversion to prevent crashes on NaN or missing data
    text = str(text)
    text = normalizer.normalize(text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r"\u200c", " ", text)
    text = re.sub(r'[؟،؛٪]', ' ', text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [word for word in my_tokenizer.tokenize_words(text) if word not in stopwords]

    # FIX: Prevent returning completely empty strings
    result = " ".join(tokens)
    return result if len(result) > 0 else " "  #because if here the last one was broken

def batch_preprocess(texts):
    return [preprocess(txt) for txt in texts]

# ===== 4. Setup Pipeline =====
tfidf = TfidfVectorizer(
    max_features=50000,
    min_df=3,
    max_df=.9,
    ngram_range=(1, 3)
)

model = LinearSVC(C=0.3)

# FIX: Define the pipeline correctly just once
base_pipeline = Pipeline([
    ('text_preprocess', FunctionTransformer(batch_preprocess)),
    ('tfidf', tfidf),
    ('Classifier', model)
])

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ===== 5. Run GridSearch =====
grid_model = GridSearchCV(
    base_pipeline,
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 3, 5],
        "tfidf__max_df": [0.8, 0.95, 1.0],
        "Classifier__C": [0.1, 1, 10]
    },
    scoring="f1_macro",
    verbose=3,
    cv=skf,
    n_jobs=-1  # FIX: Added n_jobs=-1 to use all CPU cores and speed up the heavy processing
)

grid_model.fit(X, y)
best_estimator = grid_model.best_estimator_

print("\nBest Parameters found:", grid_model.best_params_)

test_df  = pd.read_csv('test.csv', delimiter="\t")
X_test = test_df['comment']
y_test = test_df['label_id']
X_test


y_pred = best_estimator.predict(X_test)


y_pred = best_estimator.predict(X_test)

print(classification_report(y_test, y_pred))


feature_names = best_estimator.named_steps["tfidf"].get_feature_names_out()
weights = best_estimator.named_steps["Classifier"].coef_[0]
weights

importances = pd.DataFrame({
    "feature": feature_names,
    "weight": weights,
})
importances

# get new text to find label
new_comment = input("Enter Comment : ")
best_estimator.predict([new_comment])