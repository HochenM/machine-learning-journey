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
