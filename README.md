# Reinforcement Learning for Anomaly Detection in User Logon Behavior

A state-of-the-art anomaly detection system combining **Deep Q-Network (DQN) reinforcement learning**, **Isolation Forest ensemble detection**, and **sigmoid-based statistical scoring** to identify unusual user logon patterns in network security logs. The system continuously learns from analyst feedback to optimize detection thresholds and reduce false positives.

## Overview

This project implements an intelligent UEBA (User and Entity Behavior Analytics) system that:
- **Monitors user logon behavior** across 8 time intervals (3-hour windows)
- **Detects multi-dimensional anomalies** in logon times, source IPs, and destination hosts
- **Uses a DQN reinforcement learning agent** to adaptively optimize detection thresholds
- **Combines statistical + ML ensemble detection** (sigmoid scoring + Isolation Forest)
- **Differentiates behavior** across weekdays, Saturdays, and Sundays
- **Provides a CLI interface** for training, detection, model updates, and visualization

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              main.py (CLI)                  │
                    │   train | detect | update | visualize | info│
                    └──────┬──────┬──────┬──────┬──────┬─────────┘
                           │      │      │      │      │
              ┌────────────┘      │      │      │      └────────────┐
              ▼                   ▼      ▼      ▼                   ▼
     ┌────────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐
     │ data_processor  │  │ anomaly_detector │  │rl_agent (DQN)│  │visualization │
     │                │  │                  │  │              │  │              │
     │ CSV/JSON ingest│  │ Sigmoid scoring  │  │ Double DQN   │  │ Org trends   │
     │ Train/test agg │  │ Isolation Forest │  │ Replay buffer│  │ User trends  │
     │ Statistics     │  │ Multi-dimensional│  │ Threshold env│  │ Risk distrib │
     └───────┬────────┘  └────────┬─────────┘  └──────┬───────┘  └──────────────┘
             │                    │                    │
             ▼                    ▼                    ▼
     ┌────────────────┐  ┌──────────────────┐  ┌──────────────┐
     │ model_manager  │  │feedback_processor│  │   config.py  │
     │                │  │                  │  │              │
     │ Save/load JSON │  │ Feedback gen     │  │ All settings │
     │ Versioning     │  │ Model update     │  │ Dataclasses  │
     │ Atomic writes  │  │ RL integration   │  │ Logging      │
     └────────────────┘  └──────────────────┘  └──────────────┘
```

## Project Structure

```
.
├── src/                                    # Source code
│   ├── main.py                            # CLI orchestrator (entry point)
│   ├── config.py                          # Centralized configuration
│   ├── data_processor.py                  # Data ingestion & aggregation
│   ├── anomaly_detector.py                # Multi-method anomaly detection engine
│   ├── rl_agent.py                        # DQN reinforcement learning agent
│   ├── feedback_processor.py              # Feedback generation & model updates
│   ├── model_manager.py                   # Model persistence & versioning
│   ├── visualization.py                   # Unified plotting module
│   │
│   │── # Legacy modules (preserved for reference)
│   ├── User_logon_anomaly_code.py         # Original anomaly detection
│   ├── collect_train_dataFinal.py         # Original training pipeline
│   ├── collect_test_dataFinal.py          # Original test pipeline
│   ├── feedback_update_code.py            # Original feedback processor
│   ├── update_model.py                    # Original model updater
│   ├── dataAggregateRawDict.py            # Original data aggregation
│   ├── dataTestDictNew.py                 # Original test data processing
│   ├── feedback_generate.py               # Original feedback generation
│   ├── organization_trend.py              # Original org visualization
│   └── user_trend.py                      # Original user visualization
│
├── tests/                                  # Test suite
│   └── test_pipeline.py                   # 26 tests covering all modules
│
├── data/                                   # Input data files
│   ├── SBM-2023-07-05/                    # Sample log files (CSV)
│   └── destinationLabel*.csv              # Destination host mappings
├── models/                                 # Trained models (JSON)
├── outputs/                                # Detection outputs & feedback
├── logs/                                   # Application logs
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # This file
```

## How It Works

### 1. Data Collection & Training

The system analyzes historical logon data to build baseline behavior models:
- Aggregates logon events into 3-hour intervals (8 intervals per day)
- Tracks source IP addresses and destination hosts per user
- Separates patterns by day type (Weekday/Saturday/Sunday)
- Computes **real running statistics** (mean, standard deviation via Welford's algorithm)

### 2. Multi-Method Anomaly Detection

The detection engine combines two complementary approaches:

**Statistical Scoring (Sigmoid-based):**
```
Risk Score = sigmoid((x - avg) / avg_sum * multiplier) * 100
```

**Ensemble Detection (Isolation Forest):**
- Unsupervised ML model trained on baseline behavior features
- Detects anomalies that statistical methods may miss
- Combined score: `0.7 * statistical_score + 0.3 * isolation_forest_score`

**Anomaly Types Detected:**
- **Time-based**: Unusual logon times or frequencies per interval
- **Source-based**: New or anomalous source IP addresses
- **Destination-based**: New or anomalous destination hosts
- **New users**: First-time user detection with risk scoring

### 3. Deep Reinforcement Learning Threshold Optimization

Unlike the original simple threshold averaging, the system now uses a **Double DQN agent**:

**State Space** (6 dimensions):
- Mean risk score, Std risk score
- Current lower/upper thresholds
- False positive rate, Detection rate

**Action Space** (5 discrete actions):
- No change, Widen thresholds, Narrow thresholds, Shift up, Shift down

**Reward Shaping:**
| Feedback | Meaning | Reward |
|---|---|---|
| Positive | True positive (confirmed anomaly) | +1.0 |
| Negative | False positive (not anomalous) | -1.0 |
| TrueNegative | Correctly not flagged | +0.5 |
| FalseNegative | Missed anomaly | -2.0 |

**Features:**
- Experience replay buffer (10,000 experiences)
- Target network with periodic sync
- Epsilon-greedy exploration with decay
- Gradient clipping for training stability
- Automatic fallback to rule-based heuristics if PyTorch unavailable

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/rahulsingh1397/Reinforcement_learning_AnomalyDetection.git
cd Reinforcement_learning_AnomalyDetection/Final\ codes
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Core:** numpy, pandas, matplotlib, scikit-learn
**Optional:** torch (PyTorch) for DQN agent (falls back to rule-based without it)

### Usage (New CLI)

#### Check system status:
```bash
python src/main.py info
```

#### Step 1: Train the baseline model
```bash
python src/main.py train --start-date 2023-06-20 --weeks 14 --data-dir data/SBM-2023-07-05
```

#### Step 2: Detect anomalies
```bash
python src/main.py detect --date 2023-07-04 --file data/SBM-2023-07-05/SBM-2023-07-04.csv
```

#### Step 3: Update model with RL-driven feedback
```bash
python src/main.py update --date 2023-07-04
```

#### Step 4: Visualize results
```bash
python src/main.py visualize --type all
```

#### Full pipeline (detect + update + visualize):
```bash
python src/main.py pipeline --date 2023-07-04 --file data/SBM-2023-07-05/SBM-2023-07-04.csv
```

### Usage (Legacy Scripts)

The original scripts are preserved and still functional:
```bash
cd src
python collect_train_dataFinal.py    # Train
python collect_test_dataFinal.py     # Detect
python update_model.py               # Update
python organization_trend.py         # Visualize org
python user_trend.py                 # Visualize user
```

## Testing

Run the full test suite (26 tests):
```bash
python -m pytest tests/test_pipeline.py -v
```

Tests cover:
- Configuration management
- Data processing (train + test models)
- Anomaly detection (statistical + ensemble)
- RL agent (DQN + fallback)
- Feedback processing
- Model persistence
- End-to-end integration

## Data Format

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
      "0": "Positive",
      "3": "Negative"
    }
  }
]
```

## Configuration

All configuration is centralized in `src/config.py` using Python dataclasses:

| Config Class | Key Parameters | Description |
|---|---|---|
| `DataConfig` | `num_intervals=8`, `valid_logon_types=[2,3,7,9,10]` | Data ingestion settings |
| `DetectionConfig` | `default_threshold=[31,69]`, `use_isolation_forest=True` | Detection hyperparameters |
| `RLConfig` | `state_dim=6`, `action_dim=5`, `learning_rate=1e-3` | DQN agent settings |
| `FeedbackConfig` | `positive_anomaly_rate=0.6`, `random_seed=42` | Feedback simulation |
| `ModelConfig` | `enable_versioning=True`, `max_versions=10` | Model persistence |

## Improvements Over Original

| Area | Original | New |
|---|---|---|
| **RL** | Simple threshold averaging | Double DQN with experience replay |
| **Detection** | Sigmoid-only scoring | Sigmoid + Isolation Forest ensemble |
| **Statistics** | `std = 0.2 * avg` (fake) | Real running std via Welford's algorithm |
| **Architecture** | 10 loosely coupled scripts | 7 modular components + CLI |
| **Config** | Hardcoded magic numbers | Centralized dataclass config |
| **Logging** | `print()` statements | Python `logging` with file + console |
| **Error handling** | `except: pass` | Typed exceptions with context |
| **Testing** | None | 26 automated tests (pytest) |
| **Security** | `eval()` on user data | Safe JSON parsing |
| **Bugs fixed** | `dest_ffedback` typo, `.pop()` crash | All critical bugs resolved |
| **Model I/O** | No versioning | Atomic writes + versioned backups |

## Tech Stack

- **Python 3.8+** - Core language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Isolation Forest ensemble detection
- **PyTorch** - Deep Q-Network agent (optional)
- **Matplotlib** - Visualization
- **pytest** - Testing framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Authors

- **Rahul Singh** - [rahulsingh1397](https://github.com/rahulsingh1397)

## Acknowledgments

- Deep Reinforcement Learning for adaptive cybersecurity (DQN, Double DQN)
- User and Entity Behavior Analytics (UEBA) best practices
- Isolation Forest for unsupervised anomaly detection
- Designed for enterprise network security monitoring

## Contact

For questions or support, please open an issue on GitHub or contact the repository owner.

---

**Note**: This system is designed for research and educational purposes. Always validate anomaly detections with security experts before taking action in production environments.
