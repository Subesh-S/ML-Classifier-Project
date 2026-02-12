import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
from utils.feature_extractor import extract_features

test_path = r"D:\ML TRAINING\Datasets\MLDemoProj\test\test"
X_test, y_test = extract_features(test_path)

if X_test.size == 0:
    print("ERROR: Failed to extract features from test dataset. Exiting.")
    exit(1)

model = joblib.load("models/best_model.pkl")
pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
