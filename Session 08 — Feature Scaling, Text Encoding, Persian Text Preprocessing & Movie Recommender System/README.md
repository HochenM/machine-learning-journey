# Session 08 — Feature Scaling, Text Encoding, Persian Text Preprocessing & Movie Recommender System

## 🎯 What I Learned
- Applied **StandardScaler** (zero mean, unit variance) and **MinMaxScaler** (custom range) for feature scaling
- Understood **LabelEncoder** (single column encoding) vs **OneHotEncoder** (binary column encoding)
- Built **Bag of Words** models using `CountVectorizer` and **TF-IDF Vectorizer** for text feature extraction
- Learned **TF-IDF** = Term Frequency × Inverse Document Frequency (rare words = more important)
- Performed **Persian text preprocessing** using Parsivar library (normalization, tokenization, stemming)
- Built a **Content-Based Recommender System** using cosine similarity on movie metadata
- Used **NearestNeighbors** with cosine metric for efficient similarity search
- Combined movie features: Title, Genre, Director, and 4 Stars into one text field
- Handled multi-word names by replacing spaces with underscores before vectorization
- Learned that **cosine similarity** measures angle between vectors (value 1 = identical)

## 📁 Files
- `Session_8.ipynb` — Scalers, encoders, text vectorization, Persian preprocessing, recommender system
- `imdb_top_1000.csv` — IMDB Top 1000 movies dataset (1000 rows × 16 columns)
- `outputs/` — Model results and recommendations

## 🔑 Key Concepts Covered

| Concept | Method/Tool | Description |
|---------|------------|-------------|
| Standard Scaling | `StandardScaler()` | Centers data (mean=0, std=1) |
| Min-Max Scaling | `MinMaxScaler(feature_range=(0,100))` | Scales to custom range |
| Label Encoding | `LabelEncoder()` | Converts categories to integers (0,1,2...) |
| One-Hot Encoding | `OneHotEncoder()` | Creates binary columns for each category |
| Bag of Words | `CountVectorizer()` | Counts word frequencies in documents |
| TF-IDF | `TfidfVectorizer()` | Weights words by importance (rare = higher) |
| Persian Normalization | `Normalizer()` from Parsivar | Standardizes Persian text |
| Persian Tokenization | `Tokenizer()` from Parsivar | Splits text into words/sentences |
| Persian Stemming | `FindStems()` from Parsivar | Reduces words to root form |
| Cosine Similarity | `cosine_similarity()` | Measures document similarity (0 to 1) |
| Nearest Neighbors | `NearestNeighbors(metric='cosine')` | Finds k-most similar items |
| Content-Based Filtering | Manual with `cosine_similarity` | Recommends based on item features |

## 📊 Results Summary

### Movie Recommender System (Using Cosine Similarity)

| Input Movie | Top 4 Recommendations |
|-------------|----------------------|
| The Godfather | The Godfather: Part III, The Godfather: Part II, Scarface, The Irishman |
| The Dark Knight | Batman Begins, 3:10 to Yuma, The Prestige, The Dark Knight Rises |

### NearestNeighbors vs Manual Cosine Similarity

| Method | Time Complexity | Best For |
|--------|----------------|----------|
| Manual + argsort | O(n²) | Small datasets, understanding |
| NearestNeighbors | O(n log n) | Large datasets, production |

## 🧠 Key Takeaways

### Scalers
- `StandardScaler`: Good for algorithms assuming normal distribution (SVM, Linear Regression)
- `MinMaxScaler`: Good for bounded ranges, neural networks with sigmoid/tanh
- Always fit on train data, then transform both train and test

### Encoders
- `LabelEncoder`: Use for target variables (y) only
- `OneHotEncoder`: Use for categorical features (X) to avoid ordinal assumptions
- `CountVectorizer`: Converts text documents to sparse word count matrices
- `TF-IDF`: Better than CountVectorizer when common words (like 'a', 'the') should have less weight

### Persian Text Preprocessing Pipeline
1. **Normalize** — fixes different character representations (ی/ي → ی)
2. **Remove punctuation** — cleans special characters
3. **Reduce repeated chars** — "سلاااام" → "سلام"  
4. **Tokenize** — splits into words
5. **Stem** — reduces to root ("میخوانم" → "خوان")

### Recommender System
- **Cosine similarity** = 1 → identical movies, 0 → completely different
- **NearestNeighbors with brute force** = exhaustive search, accurate but slower
- **Feature engineering matters**: Combined Title + Genre + Director + 4 Stars
- **Space replacement is critical**: "Frank Darabont" → "Frank_Darabont" to keep as single token

## 🔧 Key Methods Used

```python
# StandardScaler (mean=0, std=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
original = scaler.inverse_transform(scaled_data)

# MinMaxScaler (custom range 0-100)
scaler = MinMaxScaler(feature_range=(0, 100))
scaler.fit([[9], [20]])  # Define min and max
scaled = scaler.transform(data)

# LabelEncoder (single column encoding)
encoder = LabelEncoder()
encoded = encoder.fit_transform(['mobile', 'speaker', 'mobile'])  # [1, 2, 1]

# OneHotEncoder (binary columns per category)
encoder = OneHotEncoder()
result = encoder.fit_transform(names.reshape(-1, 1)).toarray()

# CountVectorizer (Bag of Words)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
doc_matrix = vectorizer.fit_transform(['cat dog cat', 'dog bird']).toarray()
# Result: [[0, 2, 1], [1, 0, 1]]  # [bird, cat, dog]

# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs).toarray()
# Rare words get higher scores

# Persian Text Preprocessing
from parsivar import Tokenizer, Normalizer, FindStems
tokenizer = Tokenizer()
normalizer = Normalizer()
stemmer = FindStems()

def preprocess_persian(text):
    text = normalizer.normalize(text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # Remove repeated chars
    text = "".join([char for char in text if char not in punctuation_list])
    tokens = tokenizer.tokenize_words(text)
    return text, tokens

# Stemming example
stemmer.convert_to_stem('کتابی')  # → 'کتاب'
stemmer.convert_to_stem('میخوانم')  # → 'خوان'

# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([user_vector], all_vectors)

# Nearest Neighbors for Recommendations
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
nn.fit(document_vectors)
distances, indices = nn.kneighbors([user_vector])
recommendations = indices[0][1:5]  # Skip first (user's own movie)

# Content-Based Recommender (Full Pipeline)
# 1. Combine features
df["combine"] = (df["Series_Title"].str.replace(" ", "_") + " " + 
                 df['Genre'].str.replace(",", "") + " " + 
                 df['Director'].str.replace(" ", "_") + " " +
                 df['Star1'].str.replace(" ", "_") + " " + 
                 df["Star2"].str.replace(" ", "_") + " " + 
                 df['Star3'].str.replace(" ", "_") + " " +
                 df["Star4"].str.replace(" ", "_"))

# 2. Vectorize
vectorizer = CountVectorizer()
result = vectorizer.fit_transform(df['combine'])

# 3. Find similar
user_movie_index = df[df['Series_Title'] == user_movie].index[0]
user_vector = result[user_movie_index]
similarities = cosine_similarity(user_vector, result)
sorted_idx = np.argsort(similarities)[0][::-1]
recommendations = sorted_idx[1:5]
