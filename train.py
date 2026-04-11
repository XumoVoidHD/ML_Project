from __future__ import annotations

import argparse

from training.trainer import TrainingConfig, train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train battery RUL prediction models.")
    parser.add_argument("--model", choices=["xgboost", "linear", "ridge", "random_forest", "svr", "mlp", "lstm", "gru"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--min_child_weight", type=float, default=2.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=0)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--svr_c", type=float, default=10.0)
    parser.add_argument("--svr_epsilon", type=float, default=0.1)
    parser.add_argument("--svr_kernel", choices=["rbf", "linear", "poly", "sigmoid"], default="rbf")
    parser.add_argument("--svr_gamma", default="scale")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--mlp_alpha", type=float, default=1e-4)
    parser.add_argument("--mlp_learning_rate_init", type=float, default=1e-3)
    parser.add_argument("--mlp_max_iter", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        model_name=args.model,
        seed=args.seed,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        ridge_alpha=args.ridge_alpha,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        svr_c=args.svr_c,
        svr_epsilon=args.svr_epsilon,
        svr_kernel=args.svr_kernel,
        svr_gamma=args.svr_gamma,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_alpha=args.mlp_alpha,
        mlp_learning_rate_init=args.mlp_learning_rate_init,
        mlp_max_iter=args.mlp_max_iter,
    )
    output_dir = train_experiment(config)
    print(f"Saved experiment to: {output_dir}")


if __name__ == "__main__":
    main()
