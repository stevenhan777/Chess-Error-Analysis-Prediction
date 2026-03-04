"""
src/components/model_trainer.py
Trains blunder and inaccuracy models for a given username / perf_type and
saves all artifacts to the models/ directory.
"""

import sys
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from xgboost               import XGBClassifier
from catboost              import CatBoostClassifier

from src.logger    import logger
from src.exception import ChessAnalysisException
from src.utils     import save_pickle, model_base, tc_label

import pickle


MODEL_CONFIGS = {

    ("standard", "blunder"): {
        "features": [
            "time_left_sec", "time_spent_sec", "time_spent_ratio",
            "eval_unified", "complexity_material_norm", "eval_volatility_norm",
            "time_eval_volatility_int", "cumulative_time_pressure", "color_white",
        ],
        "model": XGBClassifier(
            subsample=0.7,
            scale_pos_weight=137.58139534883722,
            n_estimators=200,
            min_child_weight=3,
            max_depth=5,
            learning_rate=0.05,
            gamma=0.5,
            colsample_bytree=1.0,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        ),
    },

    ("standard", "inaccuracy"): {
        "features": [
            "increment_sec", "time_left_sec", "time_spent_sec",
            "avg_time_spent_per_move", "time_spent_ratio", "eval_unified",
            "complexity_material_norm", "eval_volatility_norm",
            "time_pressure_norm", "material_time_pressure_int",
            "time_eval_volatility_int", "move_number_norm", "late_endgame_int",
            "cumulative_time_pressure", "color_white", "time_left_ratio_clipped",
        ],
        "model": XGBClassifier(
            subsample=0.8,
            scale_pos_weight=25.997632575757574,
            n_estimators=500,
            min_child_weight=3,
            max_depth=5,
            learning_rate=0.01,
            gamma=1.0,
            colsample_bytree=1.0,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        ),
    },

    ("blitz", "blunder"): {
        "features": [
            "time_left_sec", "time_spent_ratio", "eval_unified",
            "complexity_material_norm", "eval_volatility_norm",
            "time_eval_volatility_int", "color_white",
        ],
        "model": XGBClassifier(
            subsample=0.7,
            scale_pos_weight=15.307866868381241,
            n_estimators=300,
            min_child_weight=3,
            max_depth=4,
            learning_rate=0.05,
            gamma=0,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        ),
    },

    ("blitz", "inaccuracy"): {
        "features": [
            "increment_sec", "time_left_sec", "is_king_move",
            "avg_time_spent_per_move", "time_spent_ratio", "is_mate_threat",
            "eval_unified", "complexity_material_norm", "eval_volatility_norm",
            "time_pressure_norm", "material_time_pressure_int",
            "time_eval_volatility_int", "move_number_norm",
            "cumulative_time_pressure", "color_white", "time_left_ratio_clipped",
        ],
        "model": CatBoostClassifier(
            scale_pos_weight=9.451428571428572,
            learning_rate=0.05,
            l2_leaf_reg=1,
            iterations=200,
            depth=5,
            border_count=64,
            random_seed=42,
            verbose=False,
        ),
    },
}

# Binary / flag columns that should NOT be log-transformed or scaled
_BINARY_COLS = {
    "white_castled_king", "white_castled_queen",
    "black_castled_king", "black_castled_queen",
    "queen_present", "is_middlegame", "is_pawn_move",
    "is_bishop_move", "is_rook_move", "is_queen_move",
    "is_king_move", "is_mate_threat", "is_checkmate",
    "color_white",
}


class ModelTrainer:
    """Trains and saves models for one username / perf_type combination."""

    def train_and_save(
        self,
        df:        pd.DataFrame,
        username:  str,
        perf_type: str,
    ) -> None:
        """
        Train blunder and inaccuracy models and save all artifact pickles.

        Artifacts saved per target
        --------------------------
        {base}_model.pkl
        {base}_features.pkl
        {base}_medians.pkl
        {base}_base_rate.pkl
        {base}_scaler.pkl
        {base}_log_caps.pkl
        {base}_winsor_bounds.pkl
        {base}_scale_cols.pkl
        {base}_thresholds.pkl

        where base = models/{username}_{target}_{tc}
        """
        try:
            os.makedirs("models", exist_ok=True)
            tc = tc_label(perf_type)

            for target in ("blunder", "inaccuracy"):
                config_key = (tc, target)
                if config_key not in MODEL_CONFIGS:
                    raise ValueError(f"No config defined for {config_key}")

                config   = MODEL_CONFIGS[config_key]
                features = config["features"]
                model    = config["model"]

                missing = [f for f in features if f not in df.columns]
                if missing:
                    raise ValueError(
                        f"Missing features for ({tc}, {target}): {missing}\n"
                        "Check that feature engineering ran correctly."
                    )

                target_col = f"is_{target}"
                df_clean   = df[features + [target_col]].dropna()
                X          = df_clean[features].copy()
                y          = df_clean[target_col]

                logger.info(
                    f"Training ({tc}, {target}) — "
                    f"{len(df_clean)} samples, "
                    f"positive rate: {y.mean()*100:.2f}%"
                )

                log_cols = [
                    c for c in ["increment_sec", "time_spent_ratio", "time_spent_sec"]
                    if c in features
                ]
                winsor_cols = [c for c in ["eval_unified"] if c in features]
                binary_cols = [c for c in _BINARY_COLS if c in features]
                scale_cols  = [
                    c for c in features
                    if c not in log_cols + winsor_cols + binary_cols
                ]

                log_caps = {}
                for col in log_cols:
                    cap = float(X[col].quantile(0.95))
                    log_caps[col] = cap
                    X[col] = np.log1p(X[col].clip(0, cap))

                winsor_bounds = {}
                for col in winsor_cols:
                    lo = float(X[col].quantile(0.01))
                    hi = float(X[col].quantile(0.99))
                    winsor_bounds[col] = (lo, hi)
                    X[col] = X[col].clip(lo, hi)

                all_scale_cols = log_cols + winsor_cols + scale_cols
                scaler = StandardScaler()
                X[all_scale_cols] = scaler.fit_transform(X[all_scale_cols])

                model.fit(X.to_numpy(), y.to_numpy())

                train_probs = model.predict_proba(X.to_numpy())[:, 1]
                thresholds  = {
                    "safe":   float(np.percentile(train_probs, 75)),
                    "danger": float(np.percentile(train_probs, 90)),
                }

                base      = model_base(username, target, perf_type)
                base_rate = float(y.mean())

                save_pickle(model,                f"{base}_model.pkl")
                save_pickle(features,             f"{base}_features.pkl")
                save_pickle(X.median().to_dict(), f"{base}_medians.pkl")
                save_pickle(base_rate,            f"{base}_base_rate.pkl")
                save_pickle(scaler,               f"{base}_scaler.pkl")
                save_pickle(log_caps,             f"{base}_log_caps.pkl")
                save_pickle(winsor_bounds,        f"{base}_winsor_bounds.pkl")
                save_pickle(all_scale_cols,       f"{base}_scale_cols.pkl")
                save_pickle(thresholds,           f"{base}_thresholds.pkl")

                logger.info(
                    f"  Thresholds — safe: {thresholds['safe']:.6f}  "
                    f"danger: {thresholds['danger']:.6f}"
                )

        except Exception as e:
            raise ChessAnalysisException(e, sys) from e
