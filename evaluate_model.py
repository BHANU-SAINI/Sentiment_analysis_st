import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# =============================
# 1️⃣ Load Dataset
# =============================

columns = ["target", "id", "date", "flag", "user", "text"]
df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin1",
    header=None,
    names=columns
)

print(f"✅ Data loaded: {df.shape}")
print(df.head())
print("\nColumns in dataset:", df.columns.tolist())

# Optional: sample smaller subset for speed
# df = df.sample(10000, random_state=42)

# Extract text + labels
X = df["text"]
y = df["target"].replace({0: 0, 4: 1})  # keep numeric to match model output

# =============================
# 2️⃣ Load Model and Vectorizer
# =============================
print("\n📦 Loading model and vectorizer...")
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")
print("✅ Model and vectorizer loaded successfully!")

# =============================
# 3️⃣ Transform and Predict
# =============================
print("\n🔄 Transforming text data...")
X_vec = vectorizer.transform(X)
print("✅ Transformation done.")

print("\n🧠 Making predictions...")
y_pred = model.predict(X_vec)
print("✅ Predictions complete.")

# =============================
# 4️⃣ Evaluate Model
# =============================
print("\n📊 Evaluating model performance...")

accuracy = accuracy_score(y, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:\n")
print(classification_report(y, y_pred, target_names=["Negative", "Positive"]))

# =============================
# 5️⃣ Confusion Matrix
# =============================
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Sentiment Classification")
plt.show()

# =============================
# 6️⃣ Class-wise Accuracy Bar Chart
# =============================
classes = ["Negative", "Positive"]
correct = cm.diagonal()
total = cm.sum(axis=1)
class_acc = correct / total

plt.figure(figsize=(6, 4))
plt.bar(classes, class_acc, color=["#FF6666", "#66B3FF"])
plt.title("Class-wise Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
for i, v in enumerate(class_acc):
    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontsize=11)
plt.show()

print("\n✅ Evaluation completed successfully!")
