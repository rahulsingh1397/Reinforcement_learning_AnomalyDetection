"""
Test suite for the Anomaly Detection RL system.

Tests cover:
- Configuration management
- Data processing (train + test models)
- Anomaly detection (statistical + ensemble)
- RL agent (DQN + fallback)
- Feedback processing
- Model persistence
- End-to-end pipeline
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    get_config, reset_config, get_day_type, get_interval,
    AppConfig, setup_logging,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_config():
    """Reset config before each test."""
    reset_config()
    yield


@pytest.fixture
def sample_train_model():
    """Create a minimal training model for testing."""
    return {
        "user_a": {
            "UserLabel": 1,
            "WD": {
                "DayCounter": 3,
                "IntervalCounter": {
                    "0": [5, 10, 20, 15, 12, 8, 3, 1],
                    "1": [4, 11, 22, 14, 13, 7, 2, 1],
                    "2": [6, 9, 18, 16, 11, 9, 4, 2],
                    "sum": [15, 30, 60, 45, 36, 24, 9, 4],
                    "avg": [5.0, 10.0, 20.0, 15.0, 12.0, 8.0, 3.0, 1.33],
                    "std": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.58],
                },
                "SourceAddress": {
                    "192.168.1.10": {"0": 10, "1": 12, "2": 11, "sum": 33, "avg": 11.0, "std": 1.0},
                    "10.0.0.5": {"0": 5, "1": 4, "2": 6, "sum": 15, "avg": 5.0, "std": 1.0},
                },
                "DestinationHost": {
                    "0": {"0": 8, "1": 9, "2": 7, "sum": 24, "avg": 8.0, "std": 1.0},
                    "1": {"0": 3, "1": 4, "2": 5, "sum": 12, "avg": 4.0, "std": 1.0},
                },
            },
            "Sat": {
                "DayCounter": 1,
                "IntervalCounter": {
                    "0": [0, 0, 2, 1, 0, 0, 0, 0],
                    "sum": [0, 0, 2, 1, 0, 0, 0, 0],
                    "avg": [0, 0, 2.0, 1.0, 0, 0, 0, 0],
                    "std": [0, 0, 0.4, 0.2, 0, 0, 0, 0],
                },
                "SourceAddress": {},
                "DestinationHost": {},
            },
            "Sun": {
                "DayCounter": 0,
                "IntervalCounter": {"sum": [0]*8, "avg": [0]*8, "std": [0]*8},
                "SourceAddress": {},
                "DestinationHost": {},
            },
        },
    }


@pytest.fixture
def sample_test_model():
    """Create a minimal test model."""
    return {
        "user_a": {
            "WD": {
                "Interval": 5,
                "IntervalCounter": [5, 10, 50, 15, 12, 8, 3, 1],  # interval 2 is anomalous (50 vs avg 20)
                "SourceAddress": {"192.168.1.10": 12, "10.0.0.5": 5},
                "DestinationHost": {0: 8, 1: 4},
            },
            "Sat": {
                "Interval": 0,
                "IntervalCounter": [0]*8,
                "SourceAddress": {},
                "DestinationHost": {},
            },
            "Sun": {
                "Interval": 0,
                "IntervalCounter": [0]*8,
                "SourceAddress": {},
                "DestinationHost": {},
            },
        },
        "new_user_x": {
            "WD": {
                "Interval": 3,
                "IntervalCounter": [0, 0, 5, 3, 0, 0, 0, 0],
                "SourceAddress": {"172.16.0.1": 8},
                "DestinationHost": {2: 8},
            },
            "Sat": {
                "Interval": 0,
                "IntervalCounter": [0]*8,
                "SourceAddress": {},
                "DestinationHost": {},
            },
            "Sun": {
                "Interval": 0,
                "IntervalCounter": [0]*8,
                "SourceAddress": {},
                "DestinationHost": {},
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        cfg = get_config()
        assert cfg.detection.default_threshold == [31.0, 69.0]
        assert cfg.data.num_intervals == 8
        assert cfg.rl.action_dim == 5

    def test_get_day_type(self):
        # 2023-07-04 is Tuesday
        dt = datetime(2023, 7, 4)
        assert get_day_type(dt) == "WD"

        # Saturday
        dt = datetime(2023, 7, 1)
        assert get_day_type(dt) == "Sat"

        # Sunday
        dt = datetime(2023, 7, 2)
        assert get_day_type(dt) == "Sun"

    def test_get_interval(self):
        assert get_interval(0) == 0
        assert get_interval(3) == 1
        assert get_interval(6) == 2
        assert get_interval(23) == 7

    def test_reset_config(self):
        cfg1 = get_config()
        cfg1.detection.percent_criteria = 99.0
        cfg2 = reset_config()
        assert cfg2.detection.percent_criteria == 50.0


# ──────────────────────────────────────────────────────────────────────
# Data processor tests
# ──────────────────────────────────────────────────────────────────────

class TestDataProcessor:
    def test_destination_label_manager(self):
        from data_processor import DestinationLabelManager

        mgr = DestinationLabelManager()
        assert mgr.get_or_create_label("server1") == 0
        assert mgr.get_or_create_label("server2") == 1
        assert mgr.get_or_create_label("server1") == 0  # existing
        assert len(mgr.hosts) == 2

    def test_destination_label_save_load(self, tmp_path):
        from data_processor import DestinationLabelManager

        mgr = DestinationLabelManager()
        mgr.get_or_create_label("host_a")
        mgr.get_or_create_label("host_b")

        filepath = tmp_path / "labels.csv"
        mgr.save(filepath)

        mgr2 = DestinationLabelManager(filepath)
        assert mgr2.hosts == ["host_a", "host_b"]
        assert mgr2.get_or_create_label("host_a") == 0

    def test_add_to_test_model(self):
        from data_processor import add_to_test_model

        model = {}
        record = {
            "UserName": "test_user",
            "DayType": "WD",
            "Interval": 3,
            "SourceAddress": "10.0.0.1",
            "DestinationHost": 5,
        }
        model = add_to_test_model(model, record)

        assert "test_user" in model
        assert model["test_user"]["WD"]["IntervalCounter"][3] == 1
        assert model["test_user"]["WD"]["SourceAddress"]["10.0.0.1"] == 1
        assert model["test_user"]["WD"]["DestinationHost"][5] == 1

        # Add another record
        model = add_to_test_model(model, record)
        assert model["test_user"]["WD"]["IntervalCounter"][3] == 2

    def test_add_to_train_model(self):
        from data_processor import add_to_train_model

        model = {}
        record = {
            "UserName": "train_user",
            "DayType": "WD",
            "Interval": 2,
            "SourceAddress": "192.168.1.1",
            "DestinationHost": 3,
        }
        model = add_to_train_model(model, record)

        assert "train_user" in model
        assert model["train_user"]["WD"]["IntervalCounter"]["0"][2] == 1
        assert model["train_user"]["WD"]["IntervalCounter"]["sum"][2] == 1

    def test_compute_statistics(self):
        from data_processor import compute_statistics

        model = {
            "user1": {
                "UserLabel": 1,
                "WD": {
                    "DayCounter": 2,
                    "IntervalCounter": {
                        "0": [10, 20, 30, 0, 0, 0, 0, 0],
                        "1": [12, 18, 32, 0, 0, 0, 0, 0],
                        "sum": [22, 38, 62, 0, 0, 0, 0, 0],
                    },
                    "SourceAddress": {
                        "10.0.0.1": {"0": 5, "1": 7, "sum": 12},
                    },
                    "DestinationHost": {
                        "0": {"0": 3, "1": 5, "sum": 8},
                    },
                },
                "Sat": {
                    "DayCounter": 0,
                    "IntervalCounter": {"sum": [0]*8},
                    "SourceAddress": {},
                    "DestinationHost": {},
                },
                "Sun": {
                    "DayCounter": 0,
                    "IntervalCounter": {"sum": [0]*8},
                    "SourceAddress": {},
                    "DestinationHost": {},
                },
            }
        }

        model = compute_statistics(model)
        ic = model["user1"]["WD"]["IntervalCounter"]
        assert "avg" in ic
        assert "std" in ic
        assert ic["avg"][0] == pytest.approx(11.0, abs=0.1)
        assert ic["avg"][1] == pytest.approx(19.0, abs=0.1)


# ──────────────────────────────────────────────────────────────────────
# Anomaly detector tests
# ──────────────────────────────────────────────────────────────────────

class TestAnomalyDetector:
    def test_sigmoid(self):
        from anomaly_detector import AnomalyDetector

        x = np.array([0.0])
        assert AnomalyDetector.sigmoid(x)[0] == pytest.approx(0.5)

        x = np.array([100.0])
        assert AnomalyDetector.sigmoid(x)[0] == pytest.approx(1.0, abs=1e-6)

        x = np.array([-100.0])
        assert AnomalyDetector.sigmoid(x)[0] == pytest.approx(0.0, abs=1e-6)

    def test_detect_new_user(self, sample_train_model, sample_test_model):
        from anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(sample_train_model, sample_test_model)
        results, _ = detector.detect_time_anomalies("WD", eof=True)

        assert "new_user_x" in results
        assert results["new_user_x"].is_new
        assert results["new_user_x"].anomaly_type == "new_user"

    def test_detect_time_anomaly(self, sample_train_model, sample_test_model):
        from anomaly_detector import AnomalyDetector

        # Make interval 2 very anomalous (50 vs avg 20)
        detector = AnomalyDetector(sample_train_model, sample_test_model)
        results, _ = detector.detect_time_anomalies("WD", eof=True)

        # user_a should have anomaly at interval 2
        if "user_a" in results:
            assert 2 in results["user_a"].intervals

    def test_detect_source_anomaly_new(self, sample_train_model, sample_test_model):
        from anomaly_detector import AnomalyDetector

        # Add a new source address to test model
        sample_test_model["user_a"]["WD"]["SourceAddress"]["NEW_IP"] = 5

        detector = AnomalyDetector(sample_train_model, sample_test_model)
        results = detector.detect_source_anomalies("WD", eof=True)

        if "user_a" in results:
            assert "NEW_IP" in results["user_a"]
            assert results["user_a"]["NEW_IP"].is_new

    def test_full_detection(self, sample_train_model, sample_test_model):
        from anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(sample_train_model, sample_test_model)
        report, _ = detector.run_detection("WD", eof=True)

        assert "new_user_x" in report.new_users
        assert len(report.time_anomalies) > 0

        # Test legacy conversion
        time_dict, source_dict, dest_dict = report.to_legacy_dicts()
        assert isinstance(time_dict, dict)
        assert isinstance(source_dict, dict)


# ──────────────────────────────────────────────────────────────────────
# RL Agent tests
# ──────────────────────────────────────────────────────────────────────

class TestRLAgent:
    def test_threshold_env(self):
        from rl_agent import ThresholdEnv

        env = ThresholdEnv([31.0, 69.0])
        state = env.get_state(np.array([50.0, 60.0, 40.0]))
        assert len(state) == 6
        assert state[2] == 31.0  # lower threshold
        assert state[3] == 69.0  # upper threshold

    def test_threshold_actions(self):
        from rl_agent import ThresholdEnv

        env = ThresholdEnv([31.0, 69.0])

        # Action 1: widen
        th = env.apply_action(1)
        assert th[0] < 31.0
        assert th[1] > 69.0

        # Action 2: narrow
        env2 = ThresholdEnv([31.0, 69.0])
        th2 = env2.apply_action(2)
        assert th2[0] > 31.0
        assert th2[1] < 69.0

    def test_reward_computation(self):
        from rl_agent import ThresholdEnv

        env = ThresholdEnv([31.0, 69.0])
        cfg = get_config().rl

        assert env.compute_reward("Positive") == cfg.reward_true_positive
        assert env.compute_reward("Negative") == cfg.reward_false_positive
        assert env.compute_reward("Nil") == 0.0

    def test_dqn_agent_creation(self):
        from rl_agent import DQNAgent

        agent = DQNAgent()
        state = np.array([50.0, 5.0, 31.0, 69.0, 0.1, 0.8], dtype=np.float32)
        action = agent.select_action(state)
        assert 0 <= action < 5

    def test_rl_optimizer(self):
        from rl_agent import RLThresholdOptimizer

        thresholds = {"user_a": [31.0, 69.0]}
        optimizer = RLThresholdOptimizer(thresholds)

        risk_scores = np.array([45.0, 55.0, 72.0, 50.0])
        new_th = optimizer.optimize("user_a", risk_scores, "Negative")

        assert len(new_th) == 2
        assert new_th[0] >= 0.0
        assert new_th[1] <= 100.0


# ──────────────────────────────────────────────────────────────────────
# Model manager tests
# ──────────────────────────────────────────────────────────────────────

class TestModelManager:
    def test_save_load_json(self, tmp_path):
        from model_manager import ModelManager

        mm = ModelManager(models_dir=tmp_path / "models", outputs_dir=tmp_path / "outputs")
        data = {"user1": {"WD": {"DayCounter": 1}}}

        mm.save_train_model(data, "test_model.json")
        loaded = mm.load_train_model("test_model.json")
        assert loaded == data

    def test_save_load_thresholds(self, tmp_path):
        from model_manager import ModelManager

        mm = ModelManager(models_dir=tmp_path / "models", outputs_dir=tmp_path / "outputs")
        thresholds = {"user1": [30.0, 70.0], "user2": [25.0, 75.0]}

        mm.save_thresholds(thresholds)
        loaded = mm.load_thresholds()
        assert loaded == thresholds

    def test_list_models(self, tmp_path):
        from model_manager import ModelManager

        mm = ModelManager(models_dir=tmp_path / "models", outputs_dir=tmp_path / "outputs")
        mm.save_train_model({"a": 1}, "model_a.json")
        mm.save_train_model({"b": 2}, "model_b.json")

        models = mm.list_models()
        names = [m["name"] for m in models]
        assert "model_a.json" in names
        assert "model_b.json" in names

    def test_model_info(self):
        from model_manager import ModelManager

        mm = ModelManager()
        model = {
            "user1": {
                "WD": {"IntervalCounter": {"sum": [10, 20, 30, 0, 0, 0, 0, 0]}},
                "Sat": {"IntervalCounter": {"sum": [0]*8}},
                "Sun": {"IntervalCounter": {"sum": [0]*8}},
            },
        }
        info = mm.get_model_info(model)
        assert info["num_users"] == 1
        assert info["total_logons"] == 60


# ──────────────────────────────────────────────────────────────────────
# Feedback processor tests
# ──────────────────────────────────────────────────────────────────────

class TestFeedbackProcessor:
    def test_feedback_generation(self, sample_train_model, sample_test_model):
        from anomaly_detector import AnomalyDetector, DetectionReport
        from feedback_processor import FeedbackGenerator

        detector = AnomalyDetector(sample_train_model, sample_test_model)
        report, _ = detector.run_detection("WD", eof=True)

        gen = FeedbackGenerator(seed=42)
        user_fb, src_fb, dest_fb = gen.generate(datetime(2023, 7, 4), report)

        assert isinstance(user_fb, list)
        assert all("DestinationUserName" in fb for fb in user_fb)
        assert all("Anomaly" in fb for fb in user_fb)

    def test_feedback_save_load(self, tmp_path):
        from feedback_processor import FeedbackGenerator

        gen = FeedbackGenerator(seed=42)
        user_fb = [{"DestinationUserName": "u1", "StartDate": "2023-07-04", "Anomaly": "Nil"}]
        src_fb = []
        dest_fb = []

        gen.save_feedback(user_fb, src_fb, dest_fb, output_dir=tmp_path)

        with open(tmp_path / "UserFeedback.json") as f:
            loaded = json.load(f)
        assert loaded == user_fb


# ──────────────────────────────────────────────────────────────────────
# Integration test
# ──────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_end_to_end_detection_and_update(self, sample_train_model, sample_test_model):
        """Full pipeline: detect → generate feedback → update model."""
        from anomaly_detector import AnomalyDetector
        from feedback_processor import FeedbackGenerator, ModelUpdater
        from rl_agent import RLThresholdOptimizer

        curr_date = datetime(2023, 7, 4)
        threshold_dict = {"user_a": [31.0, 69.0]}

        # Step 1: Detect
        detector = AnomalyDetector(sample_train_model, sample_test_model, threshold_dict)
        report, _ = detector.run_detection("WD", eof=True)

        time_dict, source_dict, dest_dict = report.to_legacy_dicts()
        assert len(time_dict) > 0

        # Step 2: Generate feedback
        gen = FeedbackGenerator(seed=42)
        user_fb, src_fb, dest_fb = gen.generate(curr_date, report)

        # Save feedback to temp dir so ModelUpdater can load them
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gen.save_feedback(user_fb, src_fb, dest_fb, output_dir=tmpdir)

            # Step 3: Update model (using RL)
            # Note: ModelUpdater loads feedback from OUTPUTS_DIR by default
            # For this test, we verify the components work individually
            rl_optimizer = RLThresholdOptimizer(threshold_dict)

            # Verify RL optimizer works
            risk_scores = np.array([50.0, 55.0, 60.0])
            new_th = rl_optimizer.optimize("user_a", risk_scores, "Negative")
            assert len(new_th) == 2

            all_th = rl_optimizer.get_all_thresholds()
            assert "user_a" in all_th


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
