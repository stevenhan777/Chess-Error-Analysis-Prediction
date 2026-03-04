"""
src/components/data_transformation.py
Feature engineering for chess move-level data implemented as
Transformers composed into two Pipelines.

Pipelines
---------
build_user_feature_pipeline(username)  →  Pipeline
    Filters to one user, fixes times, computes all features, drops raw cols,
    one-hot encodes color (→ color_white), drops eval/mate, runs final dropna.

build_timeline_feature_pipeline()      →  Pipeline
    No user filter, keeps 'move' and 'color' for display.
"""

import sys

import numpy as np
import pandas as pd

from sklearn.base     import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.logger    import logger
from src.exception import ChessAnalysisException


class UserFilter(BaseEstimator, TransformerMixin):
    """Keep only rows where 'username' matches the target user."""

    def __init__(self, username: str):
        self.username = username

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            uname = str(self.username).strip()
            df = df[df["username"] == uname]
            logger.info(f"UserFilter: kept {len(df)} rows for '{uname}'")
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class TimeSpentFixer(BaseEstimator, TransformerMixin):
    """
    Replicates the time-fixing block
      1. Zero out time_spent on move 1.
      2. Add increment back to time_spent for moves > 1.
      3. Clip negatives to 0.
      4. Drop rows missing time_left_cs.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()

            df.loc[df["move_number"] == 1, ["time_spent_cs", "time_spent_sec"]] = 0

            mask = df["move_number"] != 1
            df.loc[mask, "time_spent_sec"] = (
                df.loc[mask, "time_spent_sec"] + df.loc[mask, "increment_sec"]
            )
            df.loc[mask, "time_spent_cs"] = (
                df.loc[mask, "time_spent_cs"] + df.loc[mask, "increment_sec"] * 100
            )

            df.loc[df["time_spent_cs"]  < 0, "time_spent_cs"]  = 0
            df.loc[df["time_spent_sec"] < 0, "time_spent_sec"] = 0

            df = df.dropna(subset=["time_left_cs"])
            logger.info(f"TimeSpentFixer: {len(df)} rows after dropna(time_left_cs)")
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class TimeRatioBuilder(BaseEstimator, TransformerMixin):
    """
      1. Compute avg_time_spent_per_move per (game_id, username).
      2. Drop old time_spent_ratio; recompute from time_spent_sec / avg.
      3. Cap time_spent_sec to time_left_sec; recalculate ratio where capped.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()

            df["avg_time_spent_per_move"] = (
                df.groupby(["game_id", "username"])["time_spent_sec"]
                  .transform("mean")
            )

            df = df.drop("time_spent_ratio", axis=1)
            df["time_spent_ratio"] = (
                df["time_spent_sec"] / df["avg_time_spent_per_move"]
            )

            mask = df["time_spent_sec"] > df["time_left_sec"]
            df.loc[mask, "time_spent_sec"] = df.loc[mask, "time_left_sec"]
            df.loc[mask, "time_spent_ratio"] = (
                df.loc[mask, "time_spent_sec"]
                / df.loc[mask, "time_left_sec"].replace(0, np.nan)
            ).fillna(0)

            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class EvalFlagBuilder(BaseEstimator, TransformerMixin):
    """
      - is_mate_threat, is_checkmate
      - eval_loss (centipawns lost by the moving side)
      - is_inaccuracy (100-199 cp), is_mistake (200-299 cp), is_blunder (≥300 cp)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            results = []

            for _, group in df.groupby("game_id"):
                group = group.sort_values("move_number").copy()

                group["is_mate_threat"] = (
                    (group["mate"].notna()) & (group["mate"] != 0)
                ).astype(int)
                group["is_checkmate"] = (
                    group["eval"].isna() & group["mate"].isna()
                ).astype(int)

                group["prev_eval"]   = group["eval"].shift(1)
                group["eval_change"] = group["eval"] - group["prev_eval"]

                eval_loss = []
                for _, row in group.iterrows():
                    if pd.isna(row["eval_change"]):
                        eval_loss.append(None)
                    else:
                        loss = (
                            -row["eval_change"] if row["color"] == "white"
                            else row["eval_change"]
                        )
                        eval_loss.append(max(0, loss))

                group["eval_loss"] = eval_loss
                results.append(group)

            df = pd.concat(results).reset_index(drop=True)

            df["is_inaccuracy"] = (
                (df["eval_loss"] >= 100) & (df["eval_loss"] < 200)
            ).astype(int)
            df["is_mistake"] = (
                (df["eval_loss"] >= 200) & (df["eval_loss"] < 300)
            ).astype(int)
            df["is_blunder"] = (df["eval_loss"] >= 300).astype(int)

            logger.info("EvalFlagBuilder: error flags computed")
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class EvalUnifiedBuilder(BaseEstimator, TransformerMixin):
    """
    merges centipawn eval and mate scores into one continuous column using
    the full mate-band interpolation (WHITE_CAP / BLACK_CAP buffers).
    Also patches null time_spent_sec from time_spent_cs and fills
    time_spent_ratio NaNs with 0.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()

            max_eval = df["eval"].max()
            min_eval = df["eval"].min()
            max_white_mate = df["mate"].max()
            max_black_mate = df["mate"].min()

            FLOOR_BUFFER     = 300
            WHITE_CAP_BUFFER = max_eval * 0.10
            BLACK_CAP_BUFFER = abs(min_eval) * 0.10

            WHITE_CAP   = max_eval + WHITE_CAP_BUFFER
            WHITE_FLOOR = max_eval + FLOOR_BUFFER
            BLACK_CAP   = min_eval - BLACK_CAP_BUFFER
            BLACK_FLOOR = min_eval - FLOOR_BUFFER

            logger.info(
                f"EvalUnifiedBuilder — eval range: [{min_eval}, {max_eval}] | "
                f"mate range: [{max_black_mate}, {max_white_mate}]"
            )

            def unified_eval(row):
                if row["is_checkmate"]:
                    return WHITE_CAP if row["color"] == "white" else BLACK_CAP
                elif row["is_mate_threat"]:
                    m = row["mate"]
                    if m > 0:
                        return (
                            WHITE_CAP
                            - (m - 1) * (WHITE_CAP - WHITE_FLOOR)
                            / (max_white_mate - 1)
                        )
                    else:
                        return (
                            BLACK_CAP
                            + (abs(m) - 1)
                            * (abs(BLACK_CAP) - abs(BLACK_FLOOR))
                            / (abs(max_black_mate) - 1)
                        )
                else:
                    return row["eval"]

            df["eval_unified"] = df.apply(unified_eval, axis=1)

            mask = df["time_spent_sec"].isnull()
            df.loc[mask, "time_spent_sec"] = df.loc[mask, "time_spent_cs"] / 100
            df["time_spent_ratio"] = df["time_spent_ratio"].fillna(0)

            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class EvalVolatilityBuilder(BaseEstimator, TransformerMixin):
    """
    rolling std of eval_unified over the last 3 plies per game.
    Overwrites the raw eval_volatility column computed at ingestion time

    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.sort_values(["game_id", "move_number"]).reset_index(drop=True)
            vol_results = []

            for _, group in df.groupby("game_id"):
                evals = group["eval_unified"].tolist()
                vols  = [None] * len(evals)
                for i in range(len(evals)):
                    if i >= 2:
                        recent = [
                            evals[j]
                            for j in range(max(0, i - 2), i + 1)
                            if evals[j] is not None
                        ]
                        if len(recent) >= 2:
                            vols[i] = float(np.std(recent))
                vol_results.append(pd.Series(vols, index=group.index))

            df["eval_volatility"] = pd.concat(vol_results).sort_index()
            logger.info("EvalVolatilityBuilder: eval_volatility recomputed on eval_unified")
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class NormalisedFeatureBuilder(BaseEstimator, TransformerMixin):
    """
      - complexity_material_norm   (95th-pct, clipped 0-1)
      - eval_volatility_norm       (95th-pct, clipped 0-1)
      - time_left_ratio_clipped
      - time_pressure_norm
      - material_time_pressure_int
      - time_eval_volatility_int
      - move_number_norm
      - late_endgame_int
      - cumulative_time_pressure
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()

            score_95 = df["complexity_material_score"].quantile(0.95)
            df["complexity_material_norm"] = (
                (df["complexity_material_score"] / score_95).clip(0, 1)
                if score_95 > 0 else 0.0
            )

            max_vol = df["eval_volatility"].quantile(0.95)
            df["eval_volatility_norm"] = (
                (df["eval_volatility"].fillna(0) / max_vol).clip(0, 1)
                if max_vol > 0 else 0.0
            )

            df["time_left_ratio_clipped"] = df["time_left_ratio"].clip(0, 1)
            df["time_pressure_norm"]      = (1 - df["time_left_ratio_clipped"]).clip(0, 1)

            df["material_time_pressure_int"] = (
                df["complexity_material_norm"] * df["time_pressure_norm"]
            )
            df["time_eval_volatility_int"] = (
                df["time_pressure_norm"] * df["eval_volatility_norm"]
            )

            df["move_number_norm"] = df.groupby("game_id")["move_number"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1)
            )
            df["late_endgame_int"] = df["move_number_norm"] * df["is_endgame"]

            df["cumulative_time_pressure"] = (
                df.groupby(["game_id", "color"])["time_spent_ratio"]
                  .transform(lambda x: x.expanding().mean())
            )

            logger.info("NormalisedFeatureBuilder: all derived features added")
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
      - Drops raw columns no longer needed.
      - One-hot encodes 'color' via pd.get_dummies(drop_first=True, dtype=int)
        which produces a 'color_white' column
      - Drops 'eval' and 'mate'.
      - Runs df.dropna()
    """

    # These are dropped exactly as in the original data_loader
    _DROP_RAW = [
        "username", "move", "time_left_cs", "time_spent_cs",
        "best_move", "variation", "judgment",
        "prev_eval", "eval_change", "eval_loss", "time_control",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()

            cols_to_drop = [c for c in self._DROP_RAW if c in df.columns]
            df = df.drop(cols_to_drop, axis=1)

            # produces color_white (white=1, black dropped)
            df = pd.get_dummies(df, columns=["color"], drop_first=True, dtype=int)

            df = df.drop(
                [c for c in ["eval", "mate"] if c in df.columns], axis=1
            )

            df.dropna(inplace=True)

            logger.info(
                f"ColumnDropper: {len(df)} rows, {len(df.columns)} columns remaining"
            )
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e



class TimelineTimeSpentFixer(BaseEstimator, TransformerMixin):
    """Same as TimeSpentFixer but groups avg by game_id only (no username)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            df.loc[df["move_number"] == 1, ["time_spent_cs", "time_spent_sec"]] = 0
            mask = df["move_number"] != 1
            df.loc[mask, "time_spent_sec"] = (
                df.loc[mask, "time_spent_sec"] + df.loc[mask, "increment_sec"]
            )
            df.loc[mask, "time_spent_cs"] = (
                df.loc[mask, "time_spent_cs"] + df.loc[mask, "increment_sec"] * 100
            )
            df.loc[df["time_spent_cs"]  < 0, "time_spent_cs"]  = 0
            df.loc[df["time_spent_sec"] < 0, "time_spent_sec"] = 0
            df = df.dropna(subset=["time_left_cs"])
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class TimelineTimeRatioBuilder(BaseEstimator, TransformerMixin):
    """Builds time ratios grouped by game_id only (no username)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            df["avg_time_spent_per_move"] = (
                df.groupby("game_id")["time_spent_sec"].transform("mean")
            )
            if "time_spent_ratio" in df.columns:
                df = df.drop("time_spent_ratio", axis=1)
            df["time_spent_ratio"] = (
                df["time_spent_sec"] / df["avg_time_spent_per_move"]
            )
            mask = df["time_spent_sec"] > df["time_left_sec"]
            df.loc[mask, "time_spent_sec"] = df.loc[mask, "time_left_sec"]
            df.loc[mask, "time_spent_ratio"] = (
                df.loc[mask, "time_spent_sec"]
                / df.loc[mask, "time_left_sec"].replace(0, np.nan)
            ).fillna(0)
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class TimelineEvalFlagBuilder(BaseEstimator, TransformerMixin):
    """
    Eval flags for the timeline pipeline.
    Uses the same cp thresholds as the user pipeline EXCEPT:
      - is_inaccuracy: 100 ≤ loss < 300 
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            results = []
            for _, group in df.groupby("game_id"):
                group = group.sort_values("move_number").copy()
                group["is_mate_threat"] = (
                    (group["mate"].notna()) & (group["mate"] != 0)
                ).astype(int)
                group["is_checkmate"] = (
                    group["eval"].isna() & group["mate"].isna()
                ).astype(int)
                group["prev_eval"]   = group["eval"].shift(1)
                group["eval_change"] = group["eval"] - group["prev_eval"]

                eval_loss = []
                for _, row in group.iterrows():
                    if pd.isna(row["eval_change"]):
                        eval_loss.append(None)
                    else:
                        loss = (
                            -row["eval_change"] if row["color"] == "white"
                            else row["eval_change"]
                        )
                        eval_loss.append(max(0, loss))
                group["eval_loss"] = eval_loss
                results.append(group)

            df = pd.concat(results).reset_index(drop=True)
            # Timeline uses 100–300 range for inaccuracy
            df["is_inaccuracy"] = (
                (df["eval_loss"] >= 100) & (df["eval_loss"] < 300)
            ).astype(int)
            df["is_mistake"] = (
                (df["eval_loss"] >= 200) & (df["eval_loss"] < 300)
            ).astype(int)
            df["is_blunder"] = (df["eval_loss"] >= 300).astype(int)
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class TimelineEvalUnifiedBuilder(BaseEstimator, TransformerMixin):
    """
    Simplified eval_unified for a single game
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            max_eval = df["eval"].max() if df["eval"].notna().any() else 1000
            min_eval = df["eval"].min() if df["eval"].notna().any() else -1000

            WHITE_CAP = max_eval + abs(max_eval) * 0.10
            BLACK_CAP = min_eval - abs(min_eval) * 0.10

            def _unified(row):
                if row["is_checkmate"]:
                    return WHITE_CAP if row.get("color") == "white" else BLACK_CAP
                elif row["is_mate_threat"]:
                    return WHITE_CAP if row["mate"] > 0 else BLACK_CAP
                return row["eval"] if pd.notna(row["eval"]) else 0.0

            df["eval_unified"] = df.apply(_unified, axis=1)

            mask_null = df["time_spent_sec"].isnull()
            df.loc[mask_null, "time_spent_sec"] = (
                df.loc[mask_null, "time_spent_cs"] / 100
            )
            df["time_spent_ratio"] = df["time_spent_ratio"].fillna(0)
            return df
        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


class TimelineColorEncoder(BaseEstimator, TransformerMixin):
    """
    Adds color_white flag for timeline 
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["color_white"] = (df["color"] == "white").astype(int)
        return df



def build_user_feature_pipeline(username: str) -> Pipeline:
    """
    Returns a fully engineered DataFrame with
    'color_white' (from get_dummies) and no raw columns.
    """
    return Pipeline(steps=[
        ("user_filter",        UserFilter(username=username)),
        ("time_fixer",         TimeSpentFixer()),
        ("time_ratio",         TimeRatioBuilder()),
        ("eval_flags",         EvalFlagBuilder()),
        ("eval_unified",       EvalUnifiedBuilder()),
        ("eval_volatility",    EvalVolatilityBuilder()),
        ("normalised",         NormalisedFeatureBuilder()),
        ("column_dropper",     ColumnDropper()),
    ])


def build_timeline_feature_pipeline() -> Pipeline:
    """
    Keeps 'move' and 'color' for display.
    """
    return Pipeline(steps=[
        ("time_fixer",         TimelineTimeSpentFixer()),
        ("time_ratio",         TimelineTimeRatioBuilder()),
        ("eval_flags",         TimelineEvalFlagBuilder()),
        ("eval_unified",       TimelineEvalUnifiedBuilder()),
        ("eval_volatility",    EvalVolatilityBuilder()),
        ("normalised",         NormalisedFeatureBuilder()),
        ("color_encoder",      TimelineColorEncoder()),
        # No ColumnDropper — keep move, color, eval, mate for display
    ])
