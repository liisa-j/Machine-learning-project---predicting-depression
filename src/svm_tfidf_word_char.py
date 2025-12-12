import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack

start_time = time.time()

print("Loading data...")
df = pd.read_parquet("data/shorty_clean.parquet")

X = df["clean_text"].astype(str)
y = df["label_encoded"].astype(int)


print("Building WORD TF-IDF...")
tfidf_word = TfidfVectorizer(
    stop_words='english',
    analyzer='word',
    ngram_range=(1, 2),
    min_df=3,
    max_features=300000
)

X_word = tfidf_word.fit_transform(X)


print("Building CHAR TF-IDF...")
tfidf_char = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 6),
    min_df=3,
    max_features=300000
)

X_char = tfidf_char.fit_transform(X)


print("Combining features...")
X_combined = hstack([X_word, X_char])


print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)


print("Training Linear SVM...")
model = LinearSVC()
model.fit(X_train, y_train)


print("Evaluating...")
pred = model.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

print(f"Total runtime: {round((time.time() - start_time)/60, 2)} minutes")
