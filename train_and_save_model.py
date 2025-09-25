# train_and_save_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Paths
os.makedirs("models", exist_ok=True)
TRAIN_CSV = "traineee.csv"
TEST_CSV  = "testieee.csv"

# Load
train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
test_df  = pd.read_csv(TEST_CSV, encoding="utf-8")

# Optional: sample to speed up
if len(train_df) > 50000:
    train_df = train_df.sample(n=50000, random_state=42)
if len(test_df) > 10000:
    test_df = test_df.sample(n=10000, random_state=42)

X_train = train_df['reviewText'].astype(str)
y_train = train_df['sentiment'].astype(int)
X_test  = test_df['reviewText'].astype(str)
y_test  = test_df['sentiment'].astype(int)

# Vectorize
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Train
model = LogisticRegression(max_iter=200, solver='saga', n_jobs=-1)
model.fit(X_train_vec, y_train)

# Eval
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("Saved models to models/")

# Save top features for quick explainability (optional)
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
feat_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
feat_df.sort_values("coef", ascending=False).head(10).to_csv("models/top_positive_words.csv", index=False)
feat_df.sort_values("coef").head(10).to_csv("models/top_negative_words.csv", index=False)
print("Saved top feature CSVs.")
