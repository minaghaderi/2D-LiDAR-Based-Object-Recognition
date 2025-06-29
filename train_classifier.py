import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# === CONFIG ===
DATASET_PATH = "/Users/minaghaderi/Documents/Lidar_2_Project/my_project/controllers/explore_and_map/dataset"
CLASSES = ["chair", "table", "box"]
LABEL_MAP = {name: i for i, name in enumerate(CLASSES)}

# === LOAD DATA ===
X, y = [], []

for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)
    for fname in os.listdir(folder):
        if not fname.endswith('.npy'):
            continue
        pts = np.load(os.path.join(folder, fname))
        if len(pts) < 3:
            continue
        width = np.ptp(pts[:, 0])
        height = np.ptp(pts[:, 1])
        area = width * height if height else 1e-6
        density = len(pts) / area if area > 0 else 0
        features = [width, height, width / height if height else 0, density]
        X.append(features)
        y.append(LABEL_MAP[label])

X, y = np.array(X), np.array(y)
print(f"Loaded {len(X)} samples.")

# === TRAIN / TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === TRAIN RANDOM FOREST ===
clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
clf.fit(X_train, y_train)

# === EVALUATE ===
y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=CLASSES))

# === SAVE MODEL ===
joblib.dump(clf, "object_classifier.joblib")
print("✅ Model saved to object_classifier.joblib")
