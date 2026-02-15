"""
Main CLI orchestrator for the Anomaly Detection RL system.

Usage:
    python main.py train --data-dir ../data/SBM-2023-07-05 --weeks 3
    python main.py detect --date 2023-07-04 --file ../data/SBM-2023-07-05/SBM-2023-07-04.csv
    python main.py update --date 2023-07-04
    python main.py visualize --type org
    python main.py pipeline --date 2023-07-04 --file ../data/SBM-2023-07-05/SBM-2023-07-04.csv
    python main.py info
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    setup_logging, get_config, get_day_type,
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT,
)
from data_processor import (
    DestinationLabelManager,
    ingest_csv_to_train_model,
    ingest_csv_to_test_model,
    compute_statistics,
    write_batch_json,
    ingest_json_to_test_model,
)
from anomaly_detector import AnomalyDetector
from feedback_processor import FeedbackGenerator, ModelUpdater
from model_manager import ModelManager
from rl_agent import RLThresholdOptimizer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────

def cmd_train(args) -> None:
    """Train the baseline model from historical log data."""
    setup_logging()
    cfg = get_config()
    mm = ModelManager()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR / "SBM-2023-07-05"
    init_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    weeks = args.weeks

    # Load or initialize destination labels
    dest_label_path = DATA_DIR / cfg.data.dest_label_file
    dest_mgr = DestinationLabelManager(dest_label_path if dest_label_path.exists() else None)

    model = {}

    logger.info("=" * 60)
    logger.info("TRAINING: Processing %d days from %s", weeks, init_date.strftime("%Y-%m-%d"))
    logger.info("=" * 60)

    for i in range(weeks):
        new_date = init_date + timedelta(days=i)
        day_type = get_day_type(new_date)
        filename = f"SBM-{new_date.strftime('%Y-%m-%d')}.csv"
        filepath = data_dir / filename

        if not filepath.exists():
            # Try looking in data_dir directly
            filepath = DATA_DIR / filename
            if not filepath.exists():
                logger.warning("File not found: %s, skipping day %d", filepath, i)
                continue

        logger.info("Processing day %d: %s (%s)", i + 1, filename, day_type)
        model, dest_mgr = ingest_csv_to_train_model(filepath, model, dest_mgr)

        # Increment day counters for all users
        for user_key in model:
            model[user_key][day_type]["DayCounter"] += 1

    # Compute statistics (avg, std)
    model = compute_statistics(model)

    # Save outputs
    output_name = args.output or f"TrainData_{init_date.strftime('%Y%m%d')}_{weeks}d.json"
    mm.save_train_model(model, output_name)
    dest_mgr.save(DATA_DIR / "destinationLabel_new.csv")

    info = mm.get_model_info(model)
    logger.info("Training complete: %d users, %d total logons", info["num_users"], info["total_logons"])
    print(f"\n✓ Training complete: {info['num_users']} users, {info['total_logons']} logons")
    print(f"  Model saved to: models/{output_name}")


def cmd_detect(args) -> None:
    """Run anomaly detection on test data."""
    setup_logging()
    cfg = get_config()
    mm = ModelManager()

    curr_date = datetime.strptime(args.date, "%Y-%m-%d")
    day_type = get_day_type(curr_date)
    filepath = Path(args.file)

    if not filepath.exists():
        logger.error("Test file not found: %s", filepath)
        sys.exit(1)

    # Load baseline model
    if args.model:
        baseline = mm.load_train_model(args.model)
    else:
        # Try updated model first, then initial
        baseline = mm.load_train_model()
        if not baseline:
            baseline = mm.load_initial_train_model(week=1)
        if not baseline:
            logger.error("No baseline model found. Run 'train' first.")
            sys.exit(1)

    # Load thresholds
    threshold_dict = mm.load_thresholds() if not args.fresh else {}

    # Load destination labels
    dest_label_path = DATA_DIR / cfg.data.dest_label_file
    dest_mgr = DestinationLabelManager(dest_label_path if dest_label_path.exists() else None)

    logger.info("=" * 60)
    logger.info("DETECTION: %s (day_type=%s)", curr_date.strftime("%Y-%m-%d"), day_type)
    logger.info("=" * 60)

    # Ingest test data in batches
    test_model = {}
    total_lines = 0
    eof = False
    prev_interval = {}
    batch_size = cfg.data.batch_size

    while not eof:
        test_model, dest_mgr, lines_read, eof = ingest_csv_to_test_model(
            filepath, curr_date, test_model, dest_mgr,
            batch_size=batch_size, start_line=total_lines,
        )
        total_lines += lines_read
        logger.info("Total lines processed: %d", total_lines)

        # Run detection
        detector = AnomalyDetector(baseline, test_model, threshold_dict)
        report, prev_interval = detector.run_detection(day_type, prev_interval, eof)

    # Convert to legacy format for saving
    time_dict, source_dict, dest_dict = report.to_legacy_dicts()

    # Filter empty entries
    source_dict = {k: v for k, v in source_dict.items() if v}
    dest_dict = {k: v for k, v in dest_dict.items() if v}

    # Generate feedback
    fb_gen = FeedbackGenerator()
    user_fb, src_fb, dest_fb = fb_gen.generate(curr_date, report)
    fb_gen.save_feedback(user_fb, src_fb, dest_fb)

    # Save outputs
    mm.save_test_model(test_model)
    mm.save_anomalies(time_dict, source_dict, dest_dict)
    mm.save_thresholds(threshold_dict)
    dest_mgr.save(DATA_DIR / cfg.data.dest_label_file)

    print(f"\n✓ Detection complete for {curr_date.strftime('%Y-%m-%d')}")
    print(f"  Users analyzed: {len(test_model)}")
    print(f"  Time anomalies: {len(time_dict)}")
    print(f"  Source anomalies: {len(source_dict)} users")
    print(f"  Dest anomalies: {len(dest_dict)} users")
    print(f"  New users: {len(report.new_users)}")
    print(f"  Results saved to: outputs/")


def cmd_update(args) -> None:
    """Update the model based on feedback (RL-driven)."""
    setup_logging()
    mm = ModelManager()

    curr_date = datetime.strptime(args.date, "%Y-%m-%d")

    # Load models
    if args.model:
        train_model = mm.load_train_model(args.model)
    else:
        train_model = mm.load_train_model()
        if not train_model:
            train_model = mm.load_initial_train_model(week=1)

    test_model = mm.load_test_model()
    threshold_dict = mm.load_thresholds()
    time_anomaly, source_anomaly, dest_anomaly = mm.load_anomalies()

    if not train_model or not test_model:
        logger.error("Required models not found. Run 'train' and 'detect' first.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("MODEL UPDATE: %s", curr_date.strftime("%Y-%m-%d"))
    logger.info("=" * 60)

    # Initialize RL optimizer
    rl_agent_path = mm.get_rl_agent_path()
    rl_optimizer = RLThresholdOptimizer(
        threshold_dict,
        agent_path=str(rl_agent_path) if rl_agent_path.exists() else None,
    )

    # Run update
    updater = ModelUpdater(train_model, test_model, threshold_dict, rl_optimizer)
    updated_model, updated_thresholds = updater.update(
        curr_date, time_anomaly, source_anomaly, dest_anomaly
    )

    # Save outputs
    output_name = args.output or "saveTrainDataUpdated.json"
    mm.save_train_model(updated_model, output_name)
    mm.save_thresholds(updated_thresholds)
    rl_optimizer.save_agent(str(rl_agent_path))

    info = mm.get_model_info(updated_model)
    print(f"\n✓ Model updated: {info['num_users']} users")
    print(f"  RL agent saved to: models/{mm.cfg.rl_agent_file}")
    print(f"  Updated model: models/{output_name}")
    print(f"  Updated thresholds: outputs/{mm.cfg.threshold_file}")


def cmd_visualize(args) -> None:
    """Generate visualization plots."""
    setup_logging()
    mm = ModelManager()

    # Lazy import to avoid matplotlib requirement for non-viz commands
    from visualization import (
        plot_organization_trend, plot_user_trend,
        plot_detection_summary, plot_source_anomalies,
    )

    viz_type = args.type
    show = args.show

    if viz_type in ("org", "all"):
        models = {}
        for f in sorted(MODELS_DIR.glob("TrainData*.json")):
            model = mm.load_train_model(f.name)
            if model:
                models[f.stem] = model

        if models:
            path = plot_organization_trend(models, show=show)
            print(f"  Organization trend: {path}")
        else:
            print("  No training models found for organization trend.")

    if viz_type in ("user", "all"):
        models = {}
        for f in sorted(MODELS_DIR.glob("TrainData*.json"))[:2]:
            model = mm.load_train_model(f.name)
            if model:
                models[f.stem] = model

        test_model = mm.load_test_model()
        time_anomaly, _, _ = mm.load_anomalies()

        if models and time_anomaly:
            users = list(time_anomaly.keys())[:10]
            for user in users:
                path = plot_user_trend(user, models, test_model, show=show)
                if path:
                    print(f"  User trend ({user}): {path}")

    if viz_type in ("summary", "all"):
        time_a, source_a, dest_a = mm.load_anomalies()
        test_model = mm.load_test_model()
        new_users = sum(1 for v in time_a.values() if v == "New User")

        path = plot_detection_summary(
            time_anomalies=len(time_a) - new_users,
            source_anomalies=len(source_a),
            dest_anomalies=len(dest_a),
            new_users=new_users,
            total_users=len(test_model) if test_model else 0,
            show=show,
        )
        print(f"  Detection summary: {path}")

    print("\n✓ Visualization complete")


def cmd_pipeline(args) -> None:
    """Run the full pipeline: detect → update → visualize."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("FULL PIPELINE")
    logger.info("=" * 60)

    # Step 1: Detect
    print("\n── Step 1: Anomaly Detection ──")
    cmd_detect(args)

    # Step 2: Update
    print("\n── Step 2: Model Update (RL) ──")
    update_args = argparse.Namespace(
        date=args.date,
        model=args.model,
        output=None,
    )
    cmd_update(update_args)

    # Step 3: Visualize
    print("\n── Step 3: Visualization ──")
    viz_args = argparse.Namespace(type="summary", show=False)
    try:
        cmd_visualize(viz_args)
    except Exception as e:
        logger.warning("Visualization skipped: %s", e)

    print("\n✓ Full pipeline complete!")


def cmd_info(args) -> None:
    """Display system information and available models."""
    setup_logging()
    mm = ModelManager()

    print("\n" + "=" * 60)
    print("  Anomaly Detection RL System - Info")
    print("=" * 60)
    print(f"\n  Project root: {PROJECT_ROOT}")
    print(f"  Data dir:     {DATA_DIR}")
    print(f"  Models dir:   {MODELS_DIR}")
    print(f"  Outputs dir:  {OUTPUTS_DIR}")

    print("\n  Available models:")
    for m in mm.list_models():
        print(f"    {m['name']:40s} {m['size_kb']:>8.1f} KB  ({m['modified']})")

    # Show config summary
    cfg = get_config()
    print(f"\n  Detection config:")
    print(f"    Threshold:          {cfg.detection.default_threshold}")
    print(f"    Isolation Forest:   {cfg.detection.use_isolation_forest}")
    print(f"    Real std:           {cfg.detection.use_real_std}")
    print(f"\n  RL config:")
    print(f"    State dim:          {cfg.rl.state_dim}")
    print(f"    Action dim:         {cfg.rl.action_dim}")
    print(f"    Learning rate:      {cfg.rl.learning_rate}")
    print(f"    Epsilon decay:      {cfg.rl.epsilon_decay}")

    # Check dependencies
    print("\n  Dependencies:")
    deps = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
        ("torch (PyTorch)", "torch"),
    ]
    for name, module in deps:
        try:
            m = __import__(module)
            ver = getattr(m, "__version__", "?")
            print(f"    ✓ {name:20s} {ver}")
        except ImportError:
            print(f"    ✗ {name:20s} (not installed)")

    print()


# ──────────────────────────────────────────────────────────────────────
# CLI parser
# ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anomaly-detect",
        description="Reinforcement Learning-based Anomaly Detection for User Logon Behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --start-date 2023-06-20 --weeks 3
  python main.py detect --date 2023-07-04 --file ../data/SBM-2023-07-05/SBM-2023-07-04.csv
  python main.py update --date 2023-07-04
  python main.py pipeline --date 2023-07-04 --file ../data/SBM-2023-07-05/SBM-2023-07-04.csv
  python main.py visualize --type all --show
  python main.py info
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    p_train = subparsers.add_parser("train", help="Train baseline model from historical logs")
    p_train.add_argument("--data-dir", type=str, default=None, help="Directory containing CSV log files")
    p_train.add_argument("--start-date", type=str, default="2023-06-20", help="Start date (YYYY-MM-DD)")
    p_train.add_argument("--weeks", type=int, default=3, help="Number of days to process")
    p_train.add_argument("--output", type=str, default=None, help="Output model filename")
    p_train.set_defaults(func=cmd_train)

    # Detect
    p_detect = subparsers.add_parser("detect", help="Run anomaly detection on test data")
    p_detect.add_argument("--date", type=str, required=True, help="Test date (YYYY-MM-DD)")
    p_detect.add_argument("--file", type=str, required=True, help="Path to test CSV file")
    p_detect.add_argument("--model", type=str, default=None, help="Baseline model filename")
    p_detect.add_argument("--fresh", action="store_true", help="Start with fresh thresholds")
    p_detect.set_defaults(func=cmd_detect)

    # Update
    p_update = subparsers.add_parser("update", help="Update model with feedback (RL-driven)")
    p_update.add_argument("--date", type=str, required=True, help="Date of detection (YYYY-MM-DD)")
    p_update.add_argument("--model", type=str, default=None, help="Training model filename")
    p_update.add_argument("--output", type=str, default=None, help="Output model filename")
    p_update.set_defaults(func=cmd_update)

    # Visualize
    p_viz = subparsers.add_parser("visualize", help="Generate visualization plots")
    p_viz.add_argument("--type", choices=["org", "user", "summary", "all"], default="all",
                       help="Type of visualization")
    p_viz.add_argument("--show", action="store_true", help="Display plots interactively")
    p_viz.set_defaults(func=cmd_visualize)

    # Pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run full pipeline: detect → update → visualize")
    p_pipe.add_argument("--date", type=str, required=True, help="Test date (YYYY-MM-DD)")
    p_pipe.add_argument("--file", type=str, required=True, help="Path to test CSV file")
    p_pipe.add_argument("--model", type=str, default=None, help="Baseline model filename")
    p_pipe.add_argument("--fresh", action="store_true", help="Start with fresh thresholds")
    p_pipe.set_defaults(func=cmd_pipeline)

    # Info
    p_info = subparsers.add_parser("info", help="Display system information")
    p_info.set_defaults(func=cmd_info)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
