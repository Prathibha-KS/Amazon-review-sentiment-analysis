# logistic_regression_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -------------------------------
# 1. Load preprocessed data
# -------------------------------
train_df = pd.read_csv("train_preprocessed.csv")
test_df = pd.read_csv("test_preprocessed.csv")

# -------------------------------
# 2. Optional: Sample dataset for faster training
# -------------------------------
train_df = train_df.sample(n=50000, random_state=42) if len(train_df) > 50000 else train_df
test_df = test_df.sample(n=10000, random_state=42) if len(test_df) > 10000 else test_df

# -------------------------------
# 3. Features and labels
# -------------------------------
X_train = train_df['reviewText']
y_train = train_df['sentiment']
X_test = test_df['reviewText']
y_test = test_df['sentiment']

# -------------------------------
# 4. Convert text → Bag-of-Words
# -------------------------------
vectorizer = CountVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# -------------------------------
# 5. Train Logistic Regression
# -------------------------------
model = LogisticRegression(max_iter=200, solver='saga', n_jobs=-1)
model.fit(X_train_vect, y_train)

# -------------------------------
# 6. Predict and evaluate
# -------------------------------
y_pred = model.predict(X_test_vect)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 7. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 8. Class Distribution Plot
# -------------------------------
train_df['sentiment'].value_counts().plot(kind='bar', color=['red','green'])
plt.title("Training Data Sentiment Distribution")
plt.xticks([0,1], ["Negative", "Positive"], rotation=0)
plt.show()

# -------------------------------
# 9. Word Clouds
# -------------------------------
positive_text = " ".join(train_df[train_df['sentiment']==1]['reviewText'])
negative_text = " ".join(train_df[train_df['sentiment']==0]['reviewText'])

wc_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
wc_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Reviews Word Cloud")

plt.subplot(1,2,2)
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Reviews Word Cloud")
plt.show()
