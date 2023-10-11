import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('spam_dataset.csv') # Replace 'spam_dataset.csv' with your dataset file

X = data['text'] # Email text

y = data['label'] # Spam (1) or Ham (0)

tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Adjust the number of features

X = tfidf_vectorizer.fit_transform(X)

classifier = MultinomialNB()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

new_email = ["Your email text goes here"]

new_email_vectorized = tfidf_vectorizer.transform(new_email)

predicted_class = classifier.predict(new_email_vectorized)
if predicted_class[0] == 1:
print("This email is spam.")
else:
print("This email is not spam (ham).")