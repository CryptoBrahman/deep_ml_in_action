import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Create a LightGBM dataset
lgb_train = lgb.Dataset(X, label=y)

# Set LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
}

# Perform cross-validation using LightGBM
cv_results = lgb.cv(params, lgb_train, num_boost_round=100, nfold=5, stratified=True, seed=42, early_stopping_rounds=10)

# Display cross-validation results
print(f"Cross-validated logloss scores: {cv_results['multi_logloss-mean'][-1]:.4f} +/- {cv_results['multi_logloss-stdv'][-1]:.4f}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LightGBM model on the training set
lgb_model = lgb.train(params, lgb_train, num_boost_round=100)

# Evaluate the model on the testing set
y_pred = lgb_model.predict(X_test)
y_pred_class = [round(val) for val in y_pred]
accuracy = sum(y_pred_class == y_test) / len(y_test)
print(f"Accuracy on the testing set: {accuracy:.2f}")

# Now, 'lgb_model' is trained on the entire dataset (X, y) and ready for predictions.
