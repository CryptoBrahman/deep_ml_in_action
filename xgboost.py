import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Create classifier
xgb_model = XGBClassifier()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Evaluate the model
accuracy = xgb_model.score(X_test, y_test)
print(f"Accuracy on the testing set: {accuracy:.2f}")

# Perform cross-validation
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy scores: {cv_scores}")

xgb_model.fit(X, y)
