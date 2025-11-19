# Reinforcement Learning for Anomaly Detection in User Logon Behavior

A sophisticated anomaly detection system that uses reinforcement learning principles to identify unusual user logon patterns, source addresses, and destination hosts in network security logs. The system continuously learns from feedback to improve detection accuracy over time.

## 🎯 Overview

This project implements an intelligent anomaly detection system that:
- **Monitors user logon behavior** across different time intervals (3-hour windows)
- **Detects anomalous patterns** in source IP addresses and destination hosts
- **Learns from feedback** using reinforcement learning principles
- **Adapts thresholds dynamically** based on user feedback (positive/negative)
- **Differentiates behavior** across weekdays, Saturdays, and Sundays

## 🏗️ Project Structure

```
.
├── src/                                    # Source code files
│   ├── User_logon_anomaly_code.py         # Core anomaly detection algorithms
│   ├── collect_train_dataFinal.py         # Training data collection
│   ├── collect_test_dataFinal.py          # Testing data collection & anomaly detection
│   ├── feedback_update_code.py            # Reinforcement learning feedback processor
│   ├── update_model.py                    # Model update orchestrator
│   ├── dataAggregateRawDict.py            # Training data aggregation utilities
│   ├── dataTestDictNew.py                 # Test data processing utilities
│   ├── feedback_generate.py               # Feedback generation utilities
│   ├── organization_trend.py              # Organization-wide behavior analysis
│   └── user_trend.py                      # Individual user behavior visualization
├── data/                                   # Input data files
│   ├── SBM-2023-07-05/                    # Sample log files
│   └── destinationLabel*.csv              # Destination host mappings
├── models/                                 # Trained models
│   ├── TrainDataWeek_*.json               # Weekly training models
│   └── saveTrainDataUpdated*.json         # Updated models after feedback
├── outputs/                                # Detection outputs
│   ├── AnomalousUsers.json                # Detected anomalous users
│   ├── AnomalousSource.json               # Detected anomalous source IPs
│   ├── AnomalousDestination.json          # Detected anomalous destinations
│   ├── AnomalyThreshold*.json             # Dynamic threshold values
│   └── *Feedback.json                     # Feedback data
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # This file
```

## 🔬 How It Works

### 1. **Data Collection & Training**
The system analyzes historical logon data to build baseline behavior models:
- Aggregates logon events into 3-hour intervals (8 intervals per day)
- Tracks source IP addresses and destination hosts per user
- Separates patterns by day type (Weekday/Saturday/Sunday)
- Calculates statistical measures (mean, standard deviation)

### 2. **Anomaly Detection**
Uses a sigmoid-based risk scoring system:

```
Risk Score = sigmoid((x - avg) / avg_sum * multiplier) * 100
```

Where:
- `x` = current logon count
- `avg` = historical average
- `avg_sum` = total average logons per day
- Threshold: 31-69% (dynamically adjusted)

**Anomaly Types Detected:**
- **Time-based**: Unusual logon times or frequencies
- **Source-based**: New or anomalous source IP addresses
- **Destination-based**: New or anomalous destination hosts
- **New users**: First-time user detection

### 3. **Reinforcement Learning Feedback Loop**
The system improves through feedback:

```
Positive Feedback → Anomaly confirmed → No model update
Negative Feedback → False positive → Update model & adjust thresholds
```

**Adaptive Learning:**
- Updates running averages: `avg_new = (x + avg * counter) / (counter + 1)`
- Adjusts thresholds: `threshold_new = (risk_score + threshold_old) / 2`
- Handles delayed feedback with "NoFeedback" tracking

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/rahulsingh1397/Reinforcement_learning_AnomalyDetection.git
cd Reinforcement_learning_AnomalyDetection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Usage

#### Step 1: Train the Model
```bash
cd src
python collect_train_dataFinal.py
```
- Input: `Y` for first-time training, `N` to continue from existing model
- Processes log files from `data/` directory
- Outputs trained model to `models/TrainDataWeek_*.json`

#### Step 2: Test & Detect Anomalies
```bash
python collect_test_dataFinal.py
```
- Input: `Y` for first-time testing, `N` to use updated model
- Analyzes test data and detects anomalies in real-time
- Outputs:
  - `outputs/AnomalousUsers.json`
  - `outputs/AnomalousSource.json`
  - `outputs/AnomalousDestination.json`
  - `outputs/AnomalyThreshold.json`

#### Step 3: Update Model with Feedback
```bash
python update_model.py
```
- Input: `Y` for first-time update, `N` to use previous updated model
- Processes feedback from `outputs/*Feedback.json`
- Outputs updated model to `models/saveTrainDataUpdated*.json`

#### Step 4: Visualize Trends (Optional)
```bash
# Organization-wide trends
python organization_trend.py

# Individual user trends
python user_trend.py
```

## 📊 Data Format

### Input Log Format (CSV)
Expected fields in log files:
- `StartDate`: Timestamp (UTC milliseconds)
- `Name`: Event description (must contain "logged on")
- `SourceAddress`: Source IP address
- `DestinationUserName`: Username
- `DestinationHostName`: Destination host
- `DeviceCustomNumber1`: Logon type (2,3,7,9,10 are valid)

### Feedback Format (JSON)
```json
[
  {
    "DestinationUserName": "username",
    "StartDate": "2023-07-04",
    "Anomaly": {
      "0": "Positive",  // Interval index: Positive/Negative
      "3": "Negative"
    }
  }
]
```

## 🧮 Key Algorithms

### Anomaly Detection Class
```python
class anomalyDetector():
    - logonTime_anomaly()        # Real-time interval anomaly detection
    - logonTime_eof_anomaly()    # End-of-file anomaly detection
    - source_anomaly()           # Source IP anomaly detection
    - dest_anomaly()             # Destination host anomaly detection
```

### Model Update Functions
```python
- model_update()                 # Updates user interval counters
- source_update()                # Updates source address statistics
- destination_update()           # Updates destination host statistics
- trainModelUpdate()             # Main feedback processing function
```

## 📈 Performance Metrics

The system tracks:
- **Detection Accuracy**: Based on feedback (60% positive feedback assumed)
- **Threshold Adjustments**: Number of dynamic threshold updates
- **New User Detection**: Identification of first-time users
- **False Positive Rate**: Reduced through continuous learning

## 🔧 Configuration

Key parameters in the code:
- `percent_criteria = 50`: Threshold percentage for anomaly detection
- `threshold_val = [31, 69]`: Initial risk score thresholds
- `mult_fac = 100/criteria`: Score multiplication factor
- `std = 0.2 * avg`: Standard deviation calculation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👥 Authors

- **Rahul Singh** - [rahulsingh1397](https://github.com/rahulsingh1397)

## 🙏 Acknowledgments

- Inspired by reinforcement learning principles in cybersecurity
- Uses statistical anomaly detection with adaptive thresholds
- Designed for enterprise network security monitoring

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the repository owner.

---

**Note**: This system is designed for research and educational purposes. Always validate anomaly detections with security experts before taking action in production environments.
