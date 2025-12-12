import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

start = time.time()

print("Loading data...")
df = pd.read_parquet("data/shorty_clean.parquet")

X = df["clean_text"].astype(str)
y = df["label_encoded"].astype(int)

print("Vectorizing text...")
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),        
    analyzer='char',           
    min_df=5,
    max_features=300000
)



X_tfidf = tfidf.fit_transform(X)

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("Training Linear SVM...")
clf = LinearSVC()
clf.fit(X_train, y_train)

print("Evaluating...")
pred = clf.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

print(f"Total runtime: {round((time.time() - start)/60,2)} minutes")