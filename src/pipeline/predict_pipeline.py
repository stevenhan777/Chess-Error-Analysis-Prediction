"""
src/pipeline/predict_pipeline.py
Prediction utilities
"""

import sys
import os

import numpy as np
import pandas as pd

from src.logger    import logger
from src.exception import ChessAnalysisException
from src.utils     import models_available, tc_label, load_pickle, model_base

from src.components.data_ingestion      import DataIngestion
from src.components.data_transformation import build_timeline_feature_pipeline

try:
    import chess
    _CHESS_AVAILABLE = True
except ImportError:
    _CHESS_AVAILABLE = False

# Re-export so application.py can do: from src.pipeline.predict_pipeline import models_available
__all__ = ["PredictPipeline", "models_available"]


class PredictPipeline:
    """
    Prediction wrapper for one (username, perf_type) combination.
    Models are loaded lazily and cached per instance.
    """

    def __init__(self, username: str, perf_type: str):
        self.username  = username
        self.perf_type = perf_type
        self._models   = None   # loaded on first use


    def load_models(self) -> dict:
        """
        Load all artifact pickles for this (username, perf_type).
        Results are cached on the instance.
        """
        if self._models is not None:
            return self._models

        tc = tc_label(self.perf_type)
        u  = self.username
        loaded = {}

        for target in ("blunder", "inaccuracy"):
            base = model_base(u, target, self.perf_type)
            for suffix in (
                "model", "features", "medians", "base_rate",
                "scaler", "log_caps", "winsor_bounds", "scale_cols", "thresholds",
            ):
                key  = f"{target}_{suffix}" if suffix != "model" else target
                path = f"{base}_{suffix}.pkl"
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Missing model file: {path}\n"
                        f"Run TrainPipeline().run('{u}', '{self.perf_type}') first."
                    )
                loaded[key] = load_pickle(path)

        self._models = loaded
        logger.info(f"Models loaded for ({u}, {self.perf_type})")
        return loaded


    def _features_from_fen(
        self,
        fen:              str,
        time_left_sec:    float,
        initial_time_sec: float,
        feature_cols:     list,
        medians:          dict,
    ) -> pd.DataFrame:
        """
        Derive a model feature row from a FEN + time inputs.
        """
        if not _CHESS_AVAILABLE:
            raise ImportError("python-chess not installed. Run: pip install chess")

        board = chess.Board(fen)

        def count(piece_type):
            return (len(board.pieces(piece_type, chess.WHITE)) +
                    len(board.pieces(piece_type, chess.BLACK)))

        num_pawns  = count(chess.PAWN)
        num_minor  = count(chess.KNIGHT) + count(chess.BISHOP)
        num_rooks  = count(chess.ROOK)
        num_queens = count(chess.QUEEN)
        pieces_rem = len(board.piece_map())

        complexity      = 9 * num_queens + 5 * num_rooks + 3 * num_minor + num_pawns
        complexity_norm = min(complexity / 104.0, 1.0)

        tl_ratio           = time_left_sec / initial_time_sec if initial_time_sec > 0 else 0
        time_pressure_norm = 1.0 - tl_ratio

        move_number   = board.fullmove_number
        is_endgame    = int(num_queens == 0 or pieces_rem < 14)
        is_opening    = int(move_number < 15 and not is_endgame)
        is_middlegame = int(not is_opening and not is_endgame)

        white_castled_king  = int(not board.has_kingside_castling_rights(chess.WHITE))
        white_castled_queen = int(not board.has_queenside_castling_rights(chess.WHITE))
        black_castled_king  = int(not board.has_kingside_castling_rights(chess.BLACK))
        black_castled_queen = int(not board.has_queenside_castling_rights(chess.BLACK))

        move_number_norm         = min(move_number / 80.0, 1.0)
        late_endgame_int         = move_number_norm * is_endgame
        mat_time_int             = complexity_norm * time_pressure_norm
        cumulative_time_pressure = float(medians.get("cumulative_time_pressure", 0.5))

        known = {
            "time_left_sec":             time_left_sec,
            "time_left_ratio_clipped":   tl_ratio,
            "move_number_norm":          move_number_norm,
            "cumulative_time_pressure":  cumulative_time_pressure,
            "avg_time_spent_per_move":   float(medians.get("avg_time_spent_per_move", 7.0)),
            "increment_sec":             float(medians.get("increment_sec", 0.0)),
            "time_pressure_norm":        time_pressure_norm,
            "complexity_material_norm":  complexity_norm,
            "material_time_pressure_int": mat_time_int,
            "late_endgame_int":          late_endgame_int,
            "is_middlegame":             is_middlegame,
            "color_white":               int(board.turn == chess.WHITE),
            "white_castled_king":        white_castled_king,
            "white_castled_queen":       white_castled_queen,
            "black_castled_king":        black_castled_king,
            "black_castled_queen":       black_castled_queen,
            "queen_present":             int(num_queens > 0),
            "is_pawn_move":              0,
            "is_knight_move":            0,
            "is_bishop_move":            0,
            "is_rook_move":              0,
            "is_queen_move":             0,
            "is_king_move":              0,
            "is_mate_threat":            0,
            "is_checkmate":              0,
            "eval_unified":              0.0,
            "eval_volatility_norm":      0.0,
            "time_eval_volatility_int":  time_pressure_norm * 0.0,
            "time_spent_sec":            float(medians.get("time_spent_sec", 5.0)),
            "time_spent_ratio":          float(medians.get("time_spent_ratio", 1.0)),
        }

        row = {col: known.get(col, medians.get(col, 0)) for col in feature_cols}
        return pd.DataFrame([row])


    def predict_danger(
        self,
        fen:              str,
        time_left_sec:    float,
        initial_time_sec: float,
    ) -> dict:
        """
        Return blunder + inaccuracy probabilities and risk-zone labels for
        a given FEN position and time remaining.
        """
        try:
            models  = self.load_models()
            results = {}

            for target in ("blunder", "inaccuracy"):
                model         = models[target]
                features      = models[f"{target}_features"]
                medians       = models[f"{target}_medians"]
                base_rate     = models[f"{target}_base_rate"]
                scaler        = models[f"{target}_scaler"]
                log_caps      = models[f"{target}_log_caps"]
                winsor_bounds = models[f"{target}_winsor_bounds"]
                scale_cols    = models[f"{target}_scale_cols"]

                X = self._features_from_fen(
                    fen, time_left_sec, initial_time_sec, features, medians
                )

                for col, cap in log_caps.items():
                    if col in X.columns:
                        X[col] = np.log1p(X[col].clip(0, cap))
                for col, (lo, hi) in winsor_bounds.items():
                    if col in X.columns:
                        X[col] = X[col].clip(lo, hi)
                cols_to_scale = [c for c in scale_cols if c in X.columns]
                X[cols_to_scale] = scaler.transform(X[cols_to_scale])

                prob = float(model.predict_proba(X)[0][1])

                safe_thresh   = base_rate * 1.5
                danger_thresh = base_rate * 3.0
                zone = (
                    "Safe"    if prob < safe_thresh   else
                    "Caution" if prob < danger_thresh else
                    "Danger"
                )

                results[target] = {
                    "probability":   round(prob * 100, 1),
                    "base_rate":     round(base_rate * 100, 2),
                    "safe_thresh":   round(safe_thresh * 100, 2),
                    "danger_thresh": round(danger_thresh * 100, 2),
                    "zone":          zone,
                }

            results["imputed_note"] = (
                "eval_volatility imputed as 0, indicating constant eval over the "
                "last 3 moves. Actual risk may be higher in sharp/tactical positions."
            )
            return results

        except Exception as e:
            raise ChessAnalysisException(e, sys) from e

    def time_threshold_analysis(self, df: pd.DataFrame) -> dict:
        """
        For each error type find the time-pressure % at which the error rate
        first exceeds 2x the overall baseline.
        """
        try:
            df = df.copy()
            df["pressure_pct"] = (df["time_pressure_norm"] * 100).clip(0, 100)
            bins   = list(range(0, 105, 5))
            labels = [f"{i}" for i in range(0, 100, 5)]
            df["pressure_bin"] = pd.cut(
                df["pressure_pct"], bins=bins, labels=labels, include_lowest=True
            )

            results = {}
            for target, col in [("blunder", "is_blunder"), ("inaccuracy", "is_inaccuracy")]:
                if col not in df.columns:
                    continue
                baseline = df[col].mean() * 100
                rates    = df.groupby("pressure_bin", observed=True)[col].mean() * 100

                threshold_pct = None
                for bin_label, rate in rates.items():
                    if pd.notna(rate) and rate > baseline * 2:
                        threshold_pct = int(bin_label)
                        break

                results[target] = {
                    "baseline":      round(baseline, 3),
                    "threshold_pct": threshold_pct,
                    "rates":         rates,
                    "bin_labels":    labels,
                }

            return results

        except Exception as e:
            raise ChessAnalysisException(e, sys) from e

    def post_game_analysis(self, df: pd.DataFrame, game_id: str):
        """
        Analyze error patterns for a single game from the loaded DataFrame.
        Returns None if game_id not found.
        """
        try:
            game = df[df["game_id"] == game_id].copy()
            if len(game) == 0:
                return None

            game    = game.sort_values("move_number")
            results = {"game_id": game_id, "total_moves": len(game)}

            for target, col in [("blunder", "is_blunder"), ("inaccuracy", "is_inaccuracy")]:
                if col not in game.columns:
                    continue
                errors     = game[game[col] == 1]
                non_errors = game[game[col] == 0]
                count      = len(errors)

                if count == 0:
                    results[target] = {"count": 0}
                    continue

                phase_counts = {}
                for phase_col, label in [
                    ("is_opening",    "Opening"),
                    ("is_middlegame", "Middlegame"),
                    ("is_endgame",    "Endgame"),
                ]:
                    if phase_col in errors.columns:
                        phase_counts[label] = int(errors[phase_col].sum())

                feat_comparison = {}
                for feat, label in [
                    ("time_pressure_norm",       "Time pressure"),
                    ("eval_volatility_norm",     "Eval volatility"),
                    ("complexity_material_norm", "Complexity"),
                ]:
                    if feat in game.columns:
                        err_val = float(errors[feat].mean())
                        nrm_val = float(non_errors[feat].mean()) if len(non_errors) > 0 else 0.0
                        feat_comparison[label] = {
                            "at_error":  round(err_val, 3),
                            "at_normal": round(nrm_val, 3),
                            "higher":    err_val > nrm_val,
                        }

                primary = (
                    max(
                        feat_comparison,
                        key=lambda k: abs(
                            feat_comparison[k]["at_error"]
                            - feat_comparison[k]["at_normal"]
                        ),
                    )
                    if feat_comparison else "Unknown"
                )
                dominant_phase = (
                    max(phase_counts, key=phase_counts.get)
                    if phase_counts else "Unknown"
                )

                results[target] = {
                    "count":          count,
                    "avg_move":       round(float(errors["move_number"].mean()), 1),
                    "dominant_phase": dominant_phase,
                    "phase_counts":   phase_counts,
                    "feat_comparison": feat_comparison,
                    "primary_driver": primary,
                    "avg_pressure_pct": (
                        round(float(errors["time_pressure_norm"].mean()) * 100, 1)
                        if "time_pressure_norm" in errors.columns else None
                    ),
                }

            return results

        except Exception as e:
            raise ChessAnalysisException(e, sys) from e

    def game_timeline(self, game_id: str) -> pd.DataFrame:
        """
        Fetch a single Lichess game, run the timeline feature pipeline,
        then apply model predictions to every move.

        Adds columns: blunder_prob, inaccuracy_prob, blunder_zone, inaccuracy_zone.
        """
        try:
            # 1. Fetch + transform
            ingestion  = DataIngestion()
            raw_df     = ingestion.fetch_single_game(game_id)
            pipeline   = build_timeline_feature_pipeline()
            df         = pipeline.fit_transform(raw_df)
            df         = df.sort_values("move_number").reset_index(drop=True)

            # 2. Predict
            models_dict = self.load_models()

            for target in ("blunder", "inaccuracy"):
                model         = models_dict[target]
                features      = models_dict[f"{target}_features"]
                scaler        = models_dict[f"{target}_scaler"]
                log_caps      = models_dict[f"{target}_log_caps"]
                winsor_bounds = models_dict[f"{target}_winsor_bounds"]
                scale_cols    = models_dict[f"{target}_scale_cols"]
                thresholds    = models_dict[f"{target}_thresholds"]

                for col in features:
                    if col not in df.columns:
                        df[col] = 0.0

                X = df[features].copy().fillna(0)

                for col, cap in log_caps.items():
                    if col in X.columns:
                        X[col] = np.log1p(X[col].clip(0, cap))
                for col, (lo, hi) in winsor_bounds.items():
                    if col in X.columns:
                        X[col] = X[col].clip(lo, hi)
                cols_to_scale = [c for c in scale_cols if c in X.columns]
                X[cols_to_scale] = scaler.transform(X[cols_to_scale])

                probs = model.predict_proba(X.to_numpy())[:, 1]
                df[f"{target}_prob"] = probs

                zones = []
                for p in probs:
                    if p < thresholds["safe"]:
                        zones.append("Safe")
                    elif p < thresholds["danger"]:
                        zones.append("Caution")
                    else:
                        zones.append("Danger")
                df[f"{target}_zone"] = zones

            return df

        except Exception as e:
            raise ChessAnalysisException(e, sys) from e
