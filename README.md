# 🛣️ Road Condition Detection Using Bike Sensor Data

A machine learning system that classifies road surface conditions in real time using accelerometer and gyroscope data collected from a bike-mounted sensor — no special hardware needed beyond a smartphone.

---

# 📌 Overview

Traditional road monitoring relies on manual surveys that are expensive and infrequent. This project turns every bike ride into a passive road quality scan. Sensor data from a bike-mounted IMU (Inertial Measurement Unit) is fed into a trained Random Forest classifier that labels each moment of the ride as one of four conditions:

| Condition | Description |
|-----------|-------------|
| 🟢 Smooth | Well-maintained surface, low vibration |
| 🟡 Rough | Uneven/worn road, sustained moderate vibration |
| 🟠 Speed Breaker | Short controlled bump, moderate jolt |
| 🔴 Pothole | Sudden sharp drop and recovery, high-magnitude spike |

---

# 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **94.87%** |
| Cross-Validated Accuracy | **96.03% ± 0.58%** |
| Algorithm | Random Forest (300 trees) |
| Sampling Rate | 20 Hz |

**Per-class performance:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Smooth | 0.99 | 0.98 | 0.99 |
| Rough | 0.91 | 0.96 | 0.93 |
| Speed Breaker | 0.70 | 0.63 | 0.66 |
| Pothole | 0.86 | 0.75 | 0.80 |

---

# 🧠 How It Works

Raw sensor readings (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) are noisy on their own. The key to accuracy is **feature engineering** — converting raw readings into vibration-intensity patterns over a rolling 0.5-second window:

- **acc_magnitude** — total acceleration: √(x² + y² + z²)
- **gyro_magnitude** — total rotational velocity
- **rolling_mean** — average vibration over the window (smooth vs rough distinction)
- **rolling_std** — variability of vibration (spike events like potholes)
- **jerk** — rate of change of acceleration (high for sudden impacts)

These 7 engineered features + raw readings give the model enough signal to reliably distinguish a pothole from a speed breaker from a rough road.

---

## 📂 Project Structure

```
road-condition-detection/
│
├── road_condition_detection.py   # Main script — data, features, model, plots
├── confusion_matrix.png          # Model evaluation plot
├── vibration_timeline.png        # Ride timeline colored by road condition
└── README.md
```

---

# ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Killarilahari/Road-condition-detection.git
cd Road-condition-detection
```

**2. Install dependencies**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

**3. Run the script**
```bash
python road_condition_detection.py
```

> By default the script generates a synthetic dataset for demonstration.  
> To use your own sensor data, set `DATA_PATH = "your_file.csv"` at the top of the script.  
> Required columns: `time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed_kmph, label`

---

# 📈 Output Visualizations

**Confusion Matrix** — shows per-class prediction accuracy on the test set.

**Ride Timeline Chart** — plots vibration intensity over the full ride with background shading by predicted road condition. Anyone can scan it left-to-right and instantly see which road segments were bad.

**Road Quality Summary** — plain-English output showing percentage of the ride spent on each condition and an overall verdict (Good / Moderate / Poor).

```
============================================================
ROAD QUALITY SUMMARY FOR THIS RIDE
============================================================
Smooth         :  61.2%  #############################
Rough          :  31.1%  ###############
Speed Breaker  :   5.1%  ##
Pothole        :   2.6%  #

Detected Pothole Events : 154
Overall Verdict         : Moderate road quality - some rough patches detected.
```

---

# 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

---

# 🌍 Real-World Deployment

The trained model can be deployed on a **Raspberry Pi + MPU-6050 sensor** mounted to a bike for real-time, on-device classification at under 50ms per prediction cycle. With GPS tagging, data from multiple riders can be aggregated into a **city-wide road quality heat map** — crowd-sourced road monitoring with zero dedicated infrastructure.

---

# 👩‍💻 Author

**Killari Lahari**  
B.Tech – Computer Science & Engineering (AI & ML)  
📧 laharikillari007@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/lahari-killari-375587324)
