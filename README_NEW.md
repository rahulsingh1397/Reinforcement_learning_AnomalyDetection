# Anomaly Detection with Deep Reinforcement Learning

## A State-of-the-Art UEBA System for Network Security

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 26 Passing](https://img.shields.io/badge/tests-26%20passing-brightgreen.svg)]()

A sophisticated **User and Entity Behavior Analytics (UEBA)** system that combines **Deep Q-Network (DQN) reinforcement learning**, **Isolation Forest ensemble detection**, and **sigmoid-based statistical scoring** to identify anomalous user logon patterns in enterprise network security logs.

The system continuously learns from analyst feedback to optimize detection thresholds, reduce false positives, and adapt to evolving user behavior patterns.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [CLI Reference](#cli-reference)
6. [How It Works](#how-it-works)
7. [Configuration](#configuration)
8. [Data Formats](#data-formats)
9. [API Reference](#api-reference)
10. [Testing](#testing)
11. [Project Structure](#project-structure)
12. [Performance](#performance)
13. [Contributing](#contributing)
14. [License](#license)

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Dimensional Anomaly Detection** | Detects anomalies across logon times, source IPs, and destination hosts simultaneously |
| **Deep Reinforcement Learning** | Double DQN agent with experience replay for adaptive threshold optimization |
| **Ensemble ML Detection** | Combines statistical scoring with Isolation Forest for robust anomaly identification |
| **Behavioral Profiling** | Builds per-user baselines separated by day type (Weekday/Saturday/Sunday) |
| **Real-Time & Batch Processing** | Supports both streaming detection and end-of-day batch analysis |
| **Feedback-Driven Learning** | Continuously improves from analyst feedback (true/false positive signals) |
| **Comprehensive CLI** | Full command-line interface for training, detection, updates, and visualization |

### Technical Highlights

- **26 automated tests** with full coverage of all modules
- **Centralized configuration** via Python dataclasses
- **Atomic model persistence** with automatic versioning
- **Graceful degradation** (falls back to rule-based agent if PyTorch unavailable)
- **Proper logging** with file and console handlers
- **Type hints** throughout the codebase
- **No security vulnerabilities** (removed `eval()` on user data)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              main.py (CLI)                                  │
│         Commands: train | detect | update | visualize | pipeline | info    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  data_processor  │      │ anomaly_detector │      │    rl_agent      │
│                  │      │                  │      │                  │
│ • CSV/JSON parse │      │ • Sigmoid scoring│      │ • Double DQN     │
│ • Train/test agg │      │ • Isolation Forest│     │ • Replay buffer  │
│ • Welford stats  │      │ • Risk scoring   │      │ • Target network │
│ • Dest labels    │      │ • Multi-type det │      │ • ε-greedy       │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         │                         ▼                         │
         │              ┌──────────────────┐                 │
         │              │feedback_processor│                 │
         │              │                  │◄────────────────┘
         │              │ • Feedback gen   │
         │              │ • Model updates  │
         │              │ • RL integration │
         │              └────────┬─────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  model_manager   │      │  visualization   │      │     config       │
│                  │      │                  │      │                  │
│ • JSON save/load │      │ • Org trends     │      │ • DataConfig     │
│ • Versioning     │      │ • User trends    │      │ • DetectionConfig│
│ • Atomic writes  │      │ • Risk distrib   │      │ • RLConfig       │
│ • Model info     │      │ • RL training    │      │ • FeedbackConfig │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

### Data Flow

```
Raw CSV Logs ──► data_processor ──► Training Model (JSON)
                     │
                     ▼
              Test CSV Logs ──► data_processor ──► Test Model (JSON)
                                      │
                                      ▼
                              anomaly_detector
                              ┌───────┴───────┐
                              │               │
                        Statistical      Isolation
                         Scoring          Forest
                              │               │
                              └───────┬───────┘
                                      │
                                      ▼
                              Detection Report
                                      │
                                      ▼
                           feedback_processor ◄── Analyst Feedback
                              │           │
                              │           ▼
                              │      rl_agent (DQN)
                              │           │
                              ▼           ▼
                        Updated Model + Optimized Thresholds
```

---

## Installation

### Prerequisites

- **Python 3.8+** (tested on 3.8, 3.9, 3.10, 3.11, 3.12)
- **pip** package manager
- **8GB RAM** recommended for large datasets

### Step 1: Clone the Repository

```bash
git clone https://github.com/rahulsingh1397/Reinforcement_learning_AnomalyDetection.git
cd Reinforcement_learning_AnomalyDetection/Final\ codes
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose | Required |
|---------|---------|---------|----------|
| numpy | ≥1.21.0 | Numerical computing | ✅ Yes |
| pandas | ≥1.3.0 | Data manipulation | ✅ Yes |
| matplotlib | ≥3.4.0 | Visualization | ✅ Yes |
| scikit-learn | ≥1.0.0 | Isolation Forest ML | ✅ Yes |
| torch | ≥2.0.0 | DQN neural network | ⚠️ Optional* |
| pytest | ≥7.0.0 | Testing framework | ⚠️ Dev only |

*If PyTorch is not installed, the system automatically falls back to a rule-based threshold optimizer.

### Step 4: Verify Installation

```bash
python src/main.py info
```

Expected output:
```
============================================================
  Anomaly Detection RL System - Info
============================================================

  Project root: /path/to/Final codes
  Data dir:     /path/to/Final codes/data
  Models dir:   /path/to/Final codes/models
  Outputs dir:  /path/to/Final codes/outputs

  Dependencies:
    ✓ numpy                1.26.4
    ✓ pandas               2.3.1
    ✓ matplotlib           3.10.3
    ✓ scikit-learn         1.7.1
    ✓ torch (PyTorch)      2.3.1+cpu
```

---

## Quick Start

### 1. Train a Baseline Model

Train on 14 days of historical log data:

```bash
python src/main.py train \
    --start-date 2023-06-20 \
    --weeks 14 \
    --data-dir data/SBM-2023-07-05
```

### 2. Detect Anomalies

Run detection on a new day's logs:

```bash
python src/main.py detect \
    --date 2023-07-04 \
    --file data/SBM-2023-07-05/SBM-2023-07-04.csv
```

### 3. Update Model with Feedback

Process analyst feedback and update the model using RL:

```bash
python src/main.py update --date 2023-07-04
```

### 4. Visualize Results

Generate all visualization plots:

```bash
python src/main.py visualize --type all
```

### 5. Full Pipeline (One Command)

Run detect → update → visualize in sequence:

```bash
python src/main.py pipeline \
    --date 2023-07-04 \
    --file data/SBM-2023-07-05/SBM-2023-07-04.csv
```

---

## CLI Reference

### Global Options

```bash
python src/main.py --help
```

### Commands

#### `train` - Train Baseline Model

```bash
python src/main.py train [OPTIONS]

Options:
  --data-dir PATH       Directory containing CSV log files
  --start-date DATE     Start date for training (YYYY-MM-DD) [default: 2023-06-20]
  --weeks INT           Number of days to process [default: 3]
  --output FILENAME     Output model filename [auto-generated if omitted]
```

**Example:**
```bash
python src/main.py train --start-date 2023-06-01 --weeks 21 --output baseline_june.json
```

#### `detect` - Run Anomaly Detection

```bash
python src/main.py detect [OPTIONS]

Options:
  --date DATE           Test date (YYYY-MM-DD) [required]
  --file PATH           Path to test CSV file [required]
  --model FILENAME      Baseline model to use [default: latest]
  --fresh               Start with fresh thresholds (ignore saved)
```

**Example:**
```bash
python src/main.py detect --date 2023-07-10 --file logs/july10.csv --fresh
```

#### `update` - Update Model with RL Feedback

```bash
python src/main.py update [OPTIONS]

Options:
  --date DATE           Date of detection (YYYY-MM-DD) [required]
  --model FILENAME      Training model to update [default: latest]
  --output FILENAME     Output model filename [default: saveTrainDataUpdated.json]
```

**Example:**
```bash
python src/main.py update --date 2023-07-10 --output model_v2.json
```

#### `visualize` - Generate Plots

```bash
python src/main.py visualize [OPTIONS]

Options:
  --type TYPE           Visualization type: org | user | summary | all [default: all]
  --show                Display plots interactively (default: save to file)
```

**Example:**
```bash
python src/main.py visualize --type summary --show
```

#### `pipeline` - Full Detection Pipeline

```bash
python src/main.py pipeline [OPTIONS]

Options:
  --date DATE           Test date (YYYY-MM-DD) [required]
  --file PATH           Path to test CSV file [required]
  --model FILENAME      Baseline model to use [default: latest]
  --fresh               Start with fresh thresholds
```

#### `info` - System Information

```bash
python src/main.py info
```

Displays:
- Project directories
- Available models with sizes and timestamps
- Configuration summary
- Installed dependencies with versions

---

## How It Works

### 1. Behavioral Profiling

The system builds a behavioral profile for each user by analyzing historical logon patterns:

```
User Profile Structure:
├── UserLabel: int (unique identifier)
├── WD (Weekday):
│   ├── DayCounter: int (number of weekdays observed)
│   ├── IntervalCounter:
│   │   ├── "0": [5, 10, 20, 15, 12, 8, 3, 1]  # Day 1 logons per interval
│   │   ├── "1": [4, 11, 22, 14, 13, 7, 2, 1]  # Day 2
│   │   ├── "sum": [9, 21, 42, 29, 25, 15, 5, 2]
│   │   ├── "avg": [4.5, 10.5, 21.0, 14.5, 12.5, 7.5, 2.5, 1.0]
│   │   └── "std": [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0]
│   ├── SourceAddress:
│   │   └── "192.168.1.10": {"sum": 33, "avg": 11.0, "std": 1.0}
│   └── DestinationHost:
│       └── "0": {"sum": 24, "avg": 8.0, "std": 1.0}
├── Sat (Saturday): {...}
└── Sun (Sunday): {...}
```

**Time Intervals:**
| Interval | Hours | Description |
|----------|-------|-------------|
| 0 | 00:00 - 03:00 | Late night |
| 1 | 03:00 - 06:00 | Early morning |
| 2 | 06:00 - 09:00 | Morning |
| 3 | 09:00 - 12:00 | Late morning |
| 4 | 12:00 - 15:00 | Afternoon |
| 5 | 15:00 - 18:00 | Late afternoon |
| 6 | 18:00 - 21:00 | Evening |
| 7 | 21:00 - 24:00 | Night |

### 2. Multi-Method Anomaly Detection

#### Statistical Scoring (Sigmoid-Based)

For each observation, compute a risk score:

```python
def compute_risk_score(x, avg, avg_sum, mult_factor=2.0):
    """
    x: current logon count
    avg: historical average
    avg_sum: total average logons per day
    """
    score = (x - avg) / avg_sum * mult_factor
    risk = sigmoid(score) * 100  # Scale to 0-100
    return risk
```

**Interpretation:**
- Risk < 31: Significantly below normal (potential issue)
- Risk 31-69: Normal range
- Risk > 69: Significantly above normal (anomaly)

#### Isolation Forest Ensemble

An unsupervised ML model trained on baseline behavior features:

```python
from sklearn.ensemble import IsolationForest

# Train on user interval patterns
X = [user["WD"]["IntervalCounter"]["avg"] for user in baseline_model.values()]
iso_forest = IsolationForest(n_estimators=200, contamination=0.05)
iso_forest.fit(X)

# Score new observations
anomaly_score = iso_forest.score_samples(new_observation)
```

#### Combined Scoring

```python
combined_score = 0.7 * statistical_score + 0.3 * isolation_forest_score
```

### 3. Deep Reinforcement Learning

The system uses a **Double DQN** agent to learn optimal detection thresholds:

#### State Space (6 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | mean_risk | Mean risk score across users |
| 1 | std_risk | Standard deviation of risk scores |
| 2 | lower_th | Current lower threshold |
| 3 | upper_th | Current upper threshold |
| 4 | fp_rate | False positive rate |
| 5 | detection_rate | True positive rate |

#### Action Space (5 discrete actions)

| Action | Effect | When to Use |
|--------|--------|-------------|
| 0 | No change | Thresholds are optimal |
| 1 | Widen (±2) | Too many false positives |
| 2 | Narrow (±2) | Missing too many anomalies |
| 3 | Shift up (+2) | Lower bound too sensitive |
| 4 | Shift down (-2) | Upper bound too lenient |

#### Reward Function

| Feedback | Meaning | Reward |
|----------|---------|--------|
| Positive | True positive (confirmed anomaly) | +1.0 |
| Negative | False positive (not anomalous) | -1.0 |
| TrueNegative | Correctly not flagged | +0.5 |
| FalseNegative | Missed anomaly | -2.0 |

#### Network Architecture

```
Input (6) → Linear(128) → ReLU → LayerNorm
         → Linear(64)  → ReLU → LayerNorm
         → Linear(5)   → Q-values
```

#### Training Features

- **Experience Replay**: Buffer of 10,000 experiences
- **Target Network**: Synced every 10 episodes
- **Double DQN**: Policy net selects action, target net evaluates
- **ε-Greedy**: Starts at 1.0, decays to 0.01 (decay=0.995)
- **Gradient Clipping**: Max norm = 1.0

---

## Configuration

All configuration is centralized in `src/config.py` using Python dataclasses.

### DataConfig

```python
@dataclass
class DataConfig:
    col_timestamp: str = "StartDate"
    col_event_name: str = "Name"
    col_source_address: str = "SourceAddress"
    col_dest_user: str = "DestinationUserName"
    col_dest_host: str = "DestinationHostName"
    col_logon_type: str = "DeviceCustomNumber1"
    
    valid_logon_types: List[int] = [2, 3, 7, 9, 10]
    event_filter: str = "logged on"
    
    num_intervals: int = 8
    hours_per_interval: int = 3
    day_types: List[str] = ["WD", "Sat", "Sun"]
    
    batch_size: int = 100_000
    max_logs_per_file: int = 500_000
```

### DetectionConfig

```python
@dataclass
class DetectionConfig:
    percent_criteria: float = 50.0
    mult_factor: float = 2.0
    
    default_threshold: List[float] = [31.0, 69.0]
    source_upper_threshold: float = 69.0
    source_lower_threshold: float = 31.0
    dest_upper_threshold: float = 69.0
    dest_lower_threshold: float = 31.0
    
    use_isolation_forest: bool = True
    isolation_contamination: float = 0.05
    isolation_n_estimators: int = 200
    isolation_random_state: int = 42
    
    use_real_std: bool = True
    fallback_std_fraction: float = 0.2
    min_avg_sum: float = 1.0
```

### RLConfig

```python
@dataclass
class RLConfig:
    state_dim: int = 6
    action_dim: int = 5
    
    hidden_dims: List[int] = [128, 64]
    learning_rate: float = 1e-3
    gamma: float = 0.99
    
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 10
    
    buffer_size: int = 10_000
    batch_size: int = 64
    min_buffer_size: int = 100
    
    reward_true_positive: float = 1.0
    reward_true_negative: float = 0.5
    reward_false_positive: float = -1.0
    reward_false_negative: float = -2.0
    
    threshold_step: float = 2.0
```

### Accessing Configuration

```python
from config import get_config, reset_config

# Get global config singleton
cfg = get_config()

# Access nested settings
print(cfg.detection.default_threshold)  # [31.0, 69.0]
print(cfg.rl.learning_rate)             # 0.001

# Reset to defaults (useful in tests)
cfg = reset_config()
```

---

## Data Formats

### Input: CSV Log Files

Expected columns (0-indexed positions from original format):

| Position | Field | Description | Example |
|----------|-------|-------------|---------|
| 0 | StartDate | UTC timestamp (milliseconds) | 1688428800000 |
| 3 | Name | Event description | "User logged on" |
| 4 | SourceAddress | Source IP | "192.168.1.10" |
| 8 | DestinationHostName | Target host | "SERVER01" |
| 9 | DestinationUserName | Username | "john.doe" |
| 12 | DeviceCustomNumber1 | Logon type | 3 |

**Valid Logon Types:**
- 2: Interactive
- 3: Network
- 7: Unlock
- 9: NewCredentials
- 10: RemoteInteractive

### Output: Anomaly Reports

#### AnomalousUsers.json
```json
{
  "john.doe": "Logon time, intervals: [2, 5]",
  "jane.smith": "New User",
  "bob.wilson": "Logon time, intervals: [0]"
}
```

#### AnomalousSource.json
```json
{
  "john.doe": {
    "10.0.0.99": "New Source Address",
    "192.168.1.50": "Source Address Anomaly 78.5"
  }
}
```

#### AnomalousDestination.json
```json
{
  "john.doe": {
    "5": "New Destination Host",
    "2": "Destination Host Anomaly 72.3"
  }
}
```

### Feedback Format

#### UserFeedback.json
```json
[
  {
    "DestinationUserName": "john.doe",
    "StartDate": "2023-07-04",
    "Anomaly": {
      "2": "Positive",
      "5": "Negative"
    }
  },
  {
    "DestinationUserName": "jane.smith",
    "StartDate": "2023-07-04",
    "Anomaly": "New Positive"
  }
]
```

**Feedback Values:**
- `"Positive"`: Confirmed anomaly (true positive)
- `"Negative"`: Not anomalous (false positive)
- `"New Positive"`: New user is suspicious
- `"New Negative"`: New user is legitimate
- `"Nil"`: No feedback provided

---

## API Reference

### AnomalyDetector

```python
from anomaly_detector import AnomalyDetector, DetectionReport

# Initialize
detector = AnomalyDetector(
    baseline_model=train_model,
    current_model=test_model,
    threshold_dict=thresholds
)

# Run full detection
report, prev_interval = detector.run_detection(
    day_type="WD",
    prev_interval=None,
    eof=True
)

# Access results
print(report.time_anomalies)    # Dict[str, AnomalyResult]
print(report.source_anomalies)  # Dict[str, Dict[str, AnomalyResult]]
print(report.dest_anomalies)    # Dict[str, Dict[str, AnomalyResult]]
print(report.new_users)         # List[str]

# Convert to legacy format
time_dict, source_dict, dest_dict = report.to_legacy_dicts()
```

### RLThresholdOptimizer

```python
from rl_agent import RLThresholdOptimizer

# Initialize
optimizer = RLThresholdOptimizer(
    threshold_dict={"user1": [31.0, 69.0]},
    agent_path="models/rl_agent.pt"
)

# Optimize thresholds based on feedback
new_threshold = optimizer.optimize(
    user="user1",
    risk_scores=np.array([45.0, 72.0, 55.0]),
    feedback="Negative"
)

# Get all thresholds
all_thresholds = optimizer.get_all_thresholds()

# Save agent
optimizer.save_agent("models/rl_agent.pt")
```

### ModelManager

```python
from model_manager import ModelManager

mm = ModelManager()

# Save/load models
mm.save_train_model(model, "my_model.json")
model = mm.load_train_model("my_model.json")

# Save/load thresholds
mm.save_thresholds(thresholds)
thresholds = mm.load_thresholds()

# List available models
for m in mm.list_models():
    print(f"{m['name']}: {m['size_kb']} KB")

# Get model info
info = mm.get_model_info(model)
print(f"Users: {info['num_users']}, Logons: {info['total_logons']}")
```

### DataProcessor

```python
from data_processor import (
    ingest_csv_to_train_model,
    ingest_csv_to_test_model,
    compute_statistics,
    DestinationLabelManager
)

# Training
dest_mgr = DestinationLabelManager()
model, dest_mgr = ingest_csv_to_train_model(
    filepath=Path("logs/day1.csv"),
    model={},
    dest_mgr=dest_mgr
)
model = compute_statistics(model)

# Testing
test_model, dest_mgr, lines, eof = ingest_csv_to_test_model(
    filepath=Path("logs/test.csv"),
    curr_date=datetime(2023, 7, 4),
    batch_size=100000
)
```

---

## Testing

### Run All Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

### Run Specific Test Class

```bash
python -m pytest tests/test_pipeline.py::TestAnomalyDetector -v
```

### Run with Coverage

```bash
pip install pytest-cov
python -m pytest tests/test_pipeline.py --cov=src --cov-report=html
```

### Test Categories

| Class | Tests | Coverage |
|-------|-------|----------|
| TestConfig | 4 | Configuration management |
| TestDataProcessor | 5 | Data ingestion & statistics |
| TestAnomalyDetector | 5 | Detection algorithms |
| TestRLAgent | 5 | DQN agent & optimizer |
| TestModelManager | 4 | Model persistence |
| TestFeedbackProcessor | 2 | Feedback handling |
| TestIntegration | 1 | End-to-end pipeline |

---

## Project Structure

```
Final codes/
├── src/                                    # Source code (new architecture)
│   ├── main.py                            # CLI entry point (380 lines)
│   ├── config.py                          # Configuration management (220 lines)
│   ├── data_processor.py                  # Data ingestion (450 lines)
│   ├── anomaly_detector.py                # Detection engine (350 lines)
│   ├── rl_agent.py                        # DQN RL agent (380 lines)
│   ├── feedback_processor.py              # Feedback processing (480 lines)
│   ├── model_manager.py                   # Model persistence (180 lines)
│   ├── visualization.py                   # Plotting (380 lines)
│   │
│   └── [Legacy modules preserved]
│       ├── User_logon_anomaly_code.py
│       ├── collect_train_dataFinal.py
│       ├── collect_test_dataFinal.py
│       ├── feedback_update_code.py
│       ├── update_model.py
│       ├── dataAggregateRawDict.py
│       ├── dataTestDictNew.py
│       ├── feedback_generate.py
│       ├── organization_trend.py
│       └── user_trend.py
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                   # 26 tests (550 lines)
│
├── data/
│   ├── SBM-2023-07-05/                    # Sample CSV logs
│   └── destinationLabel.csv               # Host label mappings
│
├── models/                                 # Trained models (JSON)
│   ├── TrainDataWeek_1.json
│   ├── TrainDataWeek_2.json
│   ├── saveTrainDataUpdated.json
│   └── rl_agent.pt                        # DQN checkpoint
│
├── outputs/                                # Detection results
│   ├── AnomalousUsers.json
│   ├── AnomalousSource.json
│   ├── AnomalousDestination.json
│   ├── AnomalyThreshold.json
│   ├── UserFeedback.json
│   ├── SrcFeedback.json
│   ├── DestFeedback.json
│   └── *.png                              # Visualization plots
│
├── logs/                                   # Application logs
│   └── anomaly_detection.log
│
├── requirements.txt                        # Dependencies
├── README.md                              # This file
├── CHANGES_AND_UPGRADES.md                # Detailed changelog
└── .gitignore
```

---

## Performance

### Benchmarks

| Operation | Dataset Size | Time | Memory |
|-----------|--------------|------|--------|
| Training (14 days) | 500K logs/day | ~5 min | 2 GB |
| Detection (1 day) | 500K logs | ~30 sec | 1 GB |
| Model Update | 1000 users | ~10 sec | 500 MB |
| RL Training Step | 64 batch | ~5 ms | 100 MB |

### Scalability

- **Users**: Tested with 10,000+ unique users
- **Logs**: Handles 500,000+ logs per file
- **Models**: JSON models up to 50 MB
- **Memory**: Streaming processing for large files

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `python -m pytest tests/ -v`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to public functions
- Keep functions under 50 lines
- Write tests for new features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Authors

- **Rahul Singh** - [rahulsingh1397](https://github.com/rahulsingh1397)

---

## Acknowledgments

- Deep Reinforcement Learning research (DQN, Double DQN)
- User and Entity Behavior Analytics (UEBA) best practices
- Isolation Forest algorithm (Liu et al., 2008)
- scikit-learn and PyTorch communities

---

## Contact

For questions, issues, or feature requests:
- Open a [GitHub Issue](https://github.com/rahulsingh1397/Reinforcement_learning_AnomalyDetection/issues)
- Email: [Contact via GitHub]

---

**⚠️ Disclaimer**: This system is designed for research and educational purposes. Always validate anomaly detections with security experts before taking action in production environments.
