import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time


start_time = time.time()


df = pd.read_parquet("data/shorty_features.parquet")


X = df.drop(columns=['user_id', 'tweet_date', 'language', 'clean_text', 'label_encoded'])
y = df['label_encoded']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf = RandomForestClassifier(
    n_estimators=500,         
    max_depth=40,             
    min_samples_split=10,     
    min_samples_leaf=5,       
    max_features='sqrt',      
    n_jobs=-1,                
    random_state=42,
    verbose=1                 
)

print("Training Random Forest...")
rf.fit(X_train_scaled, y_train)


y_pred = rf.predict(X_test_scaled)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

end_time = time.time()
print(f"\nTotal runtime: {(end_time - start_time)/60:.2f} minutes")