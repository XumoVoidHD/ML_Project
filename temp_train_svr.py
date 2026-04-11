from __future__ import annotations

import json

from training.trainer import TrainingConfig, train_experiment


def main() -> None:
    config = TrainingConfig(
        model_name="svr",
        seed=42,
        svr_c=10.0,
        svr_epsilon=0.1,
        svr_kernel="rbf",
        svr_gamma="scale",
    )
    output_dir = train_experiment(config)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    print(f"Saved experiment to: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
