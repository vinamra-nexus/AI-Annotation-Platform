from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

texts = [

    # Positive
    "I love this product",
    "Amazing quality",
    "Very happy with purchase",
    "Good product",
    "Excellent service",
    "Fantastic experience",
    "Awesome phone",
    "Superb quality",
    "Highly recommended",
    "Best purchase ever",

    # Negative
    "Worst product ever",
    "Terrible experience",
    "Waste of money",
    "Awful product",
    "Bad quality",
    "Very disappointing",
    "Horrible service",
    "Poor performance",
    "Not recommended",
    "I regret buying this",

    # Neutral
    "It is okay",
    "Average quality",
    "Not bad",
    "Normal product",
    "Nothing special",
    "Acceptable",
    "Works as expected"
]

labels = [

    # Positive
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",

    # Negative
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",

    # Neutral
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

model = LogisticRegression()

model.fit(X, labels)

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")