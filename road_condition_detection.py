"""
Road Condition Detection Using Bike Sensor Data
================================================

What this script does
----------------------
1. Loads accelerometer + gyroscope data collected from a bike-mounted sensor
   (columns: time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed_kmph, label)
2. Cleans the data and engineers vibration-based features that actually
   correlate with road roughness (these are the features that make the
   model accurate -- raw axis values alone are noisy and not very useful).
3. Trains a Random Forest classifier to label road condition as:
   Smooth | Rough | Pothole | Speed Breaker
4. Prints an easy-to-read report (accuracy, per-class performance,
   feature importance) and saves two plots:
     - confusion_matrix.png
     - vibration_timeline.png  -> a literal "ride" chart anyone can read
       at a glance to tell which road segments were bad.
5. Has a `predict_new_window()` function so the trained model can be
   reused on a fresh window of sensor readings later.

If you have a real CSV from your own sensor logger, just change
DATA_PATH below to its location -- the column names must match the
ones described above (or update COLUMN MAP near the top).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------
# 1. LOAD DATA  (swap this block for `pd.read_csv("your_file.csv")`)
# ---------------------------------------------------------------------
DATA_PATH = None  # e.g. "bike_sensor_log.csv" -- set this to use your real file


def generate_synthetic_bike_sensor_data(n_samples=6000):
    """
    Creates a realistic synthetic dataset that mimics a bike-mounted
    accelerometer + gyroscope log, sampled at ~20 Hz.

    Road conditions and their typical vibration signatures:
      - Smooth:        low, stable vibration
      - Rough:         moderate, sustained vibration
      - Pothole:       short, very high-magnitude spike
      - Speed Breaker: short, moderate-high spike with a slower rise/fall
    """
    rows = []
    t = 0.0
    dt = 0.05  # 20 Hz sampling

    condition_cycle = ["Smooth"] * 25 + ["Rough"] * 15 + ["Smooth"] * 10 + \
                       ["Pothole"] * 3 + ["Smooth"] * 20 + ["Speed Breaker"] * 6 + \
                       ["Smooth"] * 15 + ["Rough"] * 20

    i = 0
    while len(rows) < n_samples:
        condition = condition_cycle[i % len(condition_cycle)]
        speed = np.random.normal(18, 4)
        speed = max(5, speed)

        if condition == "Smooth":
            acc_x = np.random.normal(0, 0.15)
            acc_y = np.random.normal(0, 0.15)
            acc_z = np.random.normal(9.8, 0.2)
            gyro_x = np.random.normal(0, 1.0)
            gyro_y = np.random.normal(0, 1.0)
            gyro_z = np.random.normal(0, 1.0)
        elif condition == "Rough":
            acc_x = np.random.normal(0, 0.6)
            acc_y = np.random.normal(0, 0.6)
            acc_z = np.random.normal(9.8, 1.1)
            gyro_x = np.random.normal(0, 4.0)
            gyro_y = np.random.normal(0, 4.0)
            gyro_z = np.random.normal(0, 4.0)
        elif condition == "Pothole":
            acc_x = np.random.normal(0, 2.5)
            acc_y = np.random.normal(0, 2.5)
            acc_z = np.random.normal(9.8, 4.5)
            gyro_x = np.random.normal(0, 12.0)
            gyro_y = np.random.normal(0, 12.0)
            gyro_z = np.random.normal(0, 12.0)
        else:  # Speed Breaker
            acc_x = np.random.normal(0, 1.2)
            acc_y = np.random.normal(0, 1.2)
            acc_z = np.random.normal(9.8, 2.6)
            gyro_x = np.random.normal(0, 7.0)
            gyro_y = np.random.normal(0, 7.0)
            gyro_z = np.random.normal(0, 7.0)

        rows.append([t, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed, condition])
        t += dt
        i += 1

    cols = ["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "speed_kmph", "label"]
    return pd.DataFrame(rows, columns=cols)


if DATA_PATH:
    df = pd.read_csv(DATA_PATH)
else:
    print("No DATA_PATH set -> generating a synthetic bike sensor dataset for demonstration.\n")
    df = generate_synthetic_bike_sensor_data()

print(f"Loaded {len(df)} sensor readings.")
print(df.head(), "\n")

# ---------------------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------------------
df = df.dropna().reset_index(drop=True)
df = df[(df["speed_kmph"] > 0) & (df["speed_kmph"] < 60)]  # drop unrealistic speed readings

# ---------------------------------------------------------------------
# 3. FEATURE ENGINEERING
#    Raw acc/gyro values are noisy. What actually separates a smooth
#    road from a pothole is the *vibration magnitude* and how much it
#    varies over a short rolling window. These engineered features are
#    what give the model its accuracy.
# ---------------------------------------------------------------------
WINDOW = 10  # ~0.5 second rolling window at 20 Hz

df["acc_magnitude"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
df["gyro_magnitude"] = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)

df["acc_mag_rolling_mean"] = df["acc_magnitude"].rolling(WINDOW, min_periods=1).mean()
df["acc_mag_rolling_std"] = df["acc_magnitude"].rolling(WINDOW, min_periods=1).std().fillna(0)
df["gyro_mag_rolling_mean"] = df["gyro_magnitude"].rolling(WINDOW, min_periods=1).mean()
df["gyro_mag_rolling_std"] = df["gyro_magnitude"].rolling(WINDOW, min_periods=1).std().fillna(0)

df["jerk"] = df["acc_magnitude"].diff().fillna(0)  # rate of change -> spikes for potholes

FEATURES = [
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    "speed_kmph", "acc_magnitude", "gyro_magnitude",
    "acc_mag_rolling_mean", "acc_mag_rolling_std",
    "gyro_mag_rolling_mean", "gyro_mag_rolling_std", "jerk",
]

X = df[FEATURES]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT + MODEL
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=RANDOM_STATE, stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",   # handles the fact that potholes are rare events
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# 5-fold cross-validation for a more reliable accuracy estimate
cv_scores = cross_val_score(model, X, y_encoded, cv=5)
y_pred = model.predict(X_test)

# ---------------------------------------------------------------------
# 5. RESULTS REPORT (clear, human-readable output)
# ---------------------------------------------------------------------
print("=" * 60)
print("ROAD CONDITION DETECTION - MODEL RESULTS")
print("=" * 60)
print(f"Test Accuracy        : {accuracy_score(y_test, y_pred):.2%}")
print(f"Cross-Validated Acc.  : {cv_scores.mean():.2%}  (+/- {cv_scores.std():.2%})")
print("\nPer-Class Performance:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("Top features driving the prediction:")
print(importance_df.head(6).to_string(index=False), "\n")

# ---------------------------------------------------------------------
# 6. CONFUSION MATRIX PLOT
# ---------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Condition")
plt.ylabel("Actual Condition")
plt.title("Confusion Matrix - Road Condition Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
plt.close()

# ---------------------------------------------------------------------
# 7. "AT A GLANCE" RIDE TIMELINE
#    This is the chart meant for non-technical readers: it plots the
#    vibration intensity over the ride and color-codes each predicted
#    condition, so anyone can scan it and immediately see where the
#    bad patches of road were.
# ---------------------------------------------------------------------
df["predicted_label"] = le.inverse_transform(model.predict(X))

# Smooth predicted labels with a rolling majority vote, purely for the
# visual -- this turns flickering point-by-point predictions into clean
# readable segments without changing the model's actual per-row accuracy.
label_codes = pd.Series(le.transform(df["predicted_label"]), index=df.index)
smoothed_codes = label_codes.rolling(15, min_periods=1, center=True).apply(
    lambda x: pd.Series(x).mode().iloc[0], raw=False
).astype(int)
df["display_label"] = le.inverse_transform(smoothed_codes)

# A longer smoothing window just for the visual (not used for training/prediction)
df["vibration_smooth"] = df["acc_magnitude"].rolling(40, min_periods=1, center=True).mean()

color_map = {
    "Smooth": "#2ecc71",
    "Rough": "#f1c40f",
    "Speed Breaker": "#e67e22",
    "Pothole": "#e74c3c",
}

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["time"], df["vibration_smooth"], color="#34495e", linewidth=1.3, zorder=3)

# Shade the background by predicted condition in contiguous blocks, so the
# chart reads like a single strip anyone can scan left-to-right.
labels = df["display_label"].values
times = df["time"].values
start_idx = 0
for i in range(1, len(labels) + 1):
    if i == len(labels) or labels[i] != labels[start_idx]:
        ax.axvspan(times[start_idx], times[min(i, len(labels) - 1)],
                   color=color_map[labels[start_idx]], alpha=0.3, zorder=1)
        start_idx = i

handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in color_map.values()]
ax.legend(handles, color_map.keys(), loc="upper right", title="Road Condition")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Vibration Magnitude (smoothed)")
ax.set_title("Ride Timeline - Predicted Road Condition Over Time")
plt.tight_layout()
plt.savefig("vibration_timeline.png", dpi=150)
plt.show()
plt.close()

print("Saved plots: confusion_matrix.png, vibration_timeline.png")

# ---------------------------------------------------------------------
# 8. SIMPLE ROAD QUALITY SUMMARY (the "anyone can read this" output)
# ---------------------------------------------------------------------
summary = df["predicted_label"].value_counts(normalize=True).mul(100).round(1)
print("\n" + "=" * 60)
print("ROAD QUALITY SUMMARY FOR THIS RIDE")
print("=" * 60)
for condition in ["Smooth", "Rough", "Speed Breaker", "Pothole"]:
    pct = summary.get(condition, 0.0)
    bar = "#" * int(pct // 2)
    print(f"{condition:<15}: {pct:>5.1f}%  {bar}")

pothole_count = (df["predicted_label"] == "Pothole").sum()
rough_pct = summary.get("Rough", 0.0) + summary.get("Pothole", 0.0)
if rough_pct > 40:
    verdict = "Poor road quality - frequent rough patches detected."
elif rough_pct > 15:
    verdict = "Moderate road quality - some rough patches detected."
else:
    verdict = "Good road quality - mostly smooth riding conditions."
print(f"\nDetected Pothole Events : {pothole_count}")
print(f"Overall Verdict         : {verdict}")


# ---------------------------------------------------------------------
# 9. REUSABLE PREDICTION FUNCTION FOR NEW SENSOR WINDOWS
# ---------------------------------------------------------------------
def predict_new_window(sensor_window_df):
    """
    sensor_window_df: a DataFrame with the same raw columns
    (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed_kmph)
    for a new, unseen ride segment. Returns predicted condition labels.
    """
    d = sensor_window_df.copy()
    d["acc_magnitude"] = np.sqrt(d["acc_x"]**2 + d["acc_y"]**2 + d["acc_z"]**2)
    d["gyro_magnitude"] = np.sqrt(d["gyro_x"]**2 + d["gyro_y"]**2 + d["gyro_z"]**2)
    d["acc_mag_rolling_mean"] = d["acc_magnitude"].rolling(WINDOW, min_periods=1).mean()
    d["acc_mag_rolling_std"] = d["acc_magnitude"].rolling(WINDOW, min_periods=1).std().fillna(0)
    d["gyro_mag_rolling_mean"] = d["gyro_magnitude"].rolling(WINDOW, min_periods=1).mean()
    d["gyro_mag_rolling_std"] = d["gyro_magnitude"].rolling(WINDOW, min_periods=1).std().fillna(0)
    d["jerk"] = d["acc_magnitude"].diff().fillna(0)
    preds = model.predict(d[FEATURES])
    return le.inverse_transform(preds)
