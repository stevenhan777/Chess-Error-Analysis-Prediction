"""
src/pipeline/train_pipeline.py
End-to-end training pipeline:
"""

import sys
import pandas as pd

from src.logger    import logger
from src.exception import ChessAnalysisException

from src.components.data_ingestion      import DataIngestion
from src.components.data_transformation import build_user_feature_pipeline
from src.components.model_trainer       import ModelTrainer


class TrainPipeline:
    """
    Orchestrates the full training workflow for one username / perf_type.

    Usage
    -----
    pipeline = TrainPipeline()
    df = pipeline.run(username="MagnusCarlsen", perf_type="blitz", max_games=500)
    # df is the engineered DataFrame (also stored in global app state)
    """

    def run(
        self,
        username:  str,
        perf_type: str,
        max_games: int = 5000,
    ) -> pd.DataFrame:
        """
        Fetch, transform, train, and save.

        Returns
        -------
        pd.DataFrame  The fully-engineered move-level DataFrame
                      (same object that is passed to the dashboard).
        """
        try:
            logger.info(
                f"TrainPipeline.run — username={username}, "
                f"perf_type={perf_type}, max_games={max_games}"
            )

            # 1. Fetch raw move-level data from Lichess
            ingestion = DataIngestion()
            raw_df    = ingestion.fetch_user_games(username, perf_type, max_games)
            logger.info(f"Raw DataFrame: {raw_df.shape}")

            # 2. Feature engineering (sklearn Pipeline)
            feat_pipeline = build_user_feature_pipeline(username)
            df = feat_pipeline.fit_transform(raw_df)
            logger.info(f"Engineered DataFrame: {df.shape}")

            # 3. Train + save models
            trainer = ModelTrainer()
            trainer.train_and_save(df, username, perf_type)

            logger.info("TrainPipeline complete.")
            return df

        except Exception as e:
            raise ChessAnalysisException(e, sys) from e
