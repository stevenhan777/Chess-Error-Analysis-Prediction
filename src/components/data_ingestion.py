"""
src/components/data_ingestion.py
Fetches games from the Lichess API and converts them to a raw move-level
DataFrame.  No feature engineering happens here, that is in
data_transformation.py.

"""

import os
import sys
import json

import numpy as np
import pandas as pd
import requests

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.logger    import logger
from src.exception import ChessAnalysisException


class DataIngestion:
    """Wraps all Lichess API calls and raw move-level parsing."""


    def fetch_user_games(
        self,
        username:  str,
        perf_type: str,
        max_games: int,
    ) -> pd.DataFrame:
        """
        Fetch up to *max_games* games for *username* and return a raw
        move-level DataFrame (no feature engineering applied yet).

        Parameters
        ----------
        username  : Lichess username
        perf_type : 'blitz' or anything else → ['rapid', 'classical']
        max_games : maximum number of games to request from the API
        """
        try:
            perf_types = ["blitz"] if perf_type == "blitz" else ["rapid", "classical"]
            logger.info(
                f"Fetching up to {max_games} games for {username} "
                f"({', '.join(perf_types)})…"
            )

            raw_games = self._fetch_games(username, perf_types, max_games)

            filtered = [
                g for g in raw_games
                if "analysis" in g and len(g.get("analysis", [])) > 0
            ]
            logger.info(
                f"Total fetched: {len(raw_games)} | With analysis: {len(filtered)}"
            )

            if not filtered:
                raise ValueError(
                    "No games with engine analysis found. "
                    "Please analyse some games on Lichess first."
                )

            return self._to_move_level(filtered)

        except Exception as e:
            raise ChessAnalysisException(e, sys) from e

    def fetch_single_game(self, game_id: str) -> pd.DataFrame:
        """
        Fetch one Lichess game by ID and return a raw move-level DataFrame.
        The game must already have computer analysis on Lichess.
        """
        try:
            url = f"https://lichess.org/game/export/{game_id}"
            params  = {"pgnInJson": "true", "clocks": "true",
                       "evals": "true", "accuracy": "true"}
            headers = {"Accept": "application/json"}

            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 404:
                raise ValueError(f"Game '{game_id}' not found on Lichess.")
            if resp.status_code != 200:
                raise ConnectionError(
                    f"Lichess API returned {resp.status_code}"
                )

            game = resp.json()

            if "analysis" not in game or len(game.get("analysis", [])) == 0:
                raise ValueError(
                    "This game has no computer analysis. "
                    "Please run computer analysis on Lichess first, then retry."
                )

            return self._to_move_level([game])

        except Exception as e:
            raise ChessAnalysisException(e, sys) from e


    def _fetch_games(
        self,
        username:   str,
        perf_types: list,
        max_games:  int,
    ) -> list:
        """Call the Lichess NDJSON games endpoint and return a list of dicts."""
        url = f"https://lichess.org/api/games/user/{username}"
        params = {
            "max":       max_games,
            "perfType":  ",".join(perf_types),
            "pgnInJson": "true",
            "clocks":    "true",
            "evals":     "true",
            "opening":   "true",
            "accuracy":  "true",
        }
        headers = {"Accept": "application/x-ndjson"}

        resp = requests.get(url, headers=headers, params=params, timeout=120)
        if resp.status_code != 200:
            raise ConnectionError(
                f"Lichess API returned {resp.status_code}: {resp.text[:200]}"
            )

        games = []
        for line in resp.content.decode("utf-8").strip().split("\n"):
            line = line.strip()
            if line:
                games.append(json.loads(line))
        return games

    def _track_board_state(self, moves_san: list) -> list:
        """
        Walk through the move list in SAN and return one board-state dict
        per move.  Tracks piece counts, castling flags, and game phase.
        """
        board_states = []
        pieces_remaining = 32
        num_pawns, num_minor_pieces, num_rooks, num_queens = 16, 8, 4, 2
        wck = wcq = bck = bcq = False

        for i, move in enumerate(moves_san):
            color = "white" if i % 2 == 0 else "black"

            if move.startswith("O-O-O"):
                if color == "white": wcq = True
                else:                bcq = True
            elif move.startswith("O-O"):
                if color == "white": wck = True
                else:                bck = True

            prom_q = "=Q" in move
            prom_r = "=R" in move
            prom_m = "=B" in move or "=N" in move

            if "x" in move:
                pieces_remaining = max(0, pieces_remaining - 1)
                clean = (
                    move.replace("+", "").replace("#", "")
                        .replace("=Q", "").replace("=R", "")
                        .replace("=B", "").replace("=N", "")
                )
                first = clean[0]
                if   first.islower():      num_pawns        = max(0, num_pawns        - 1)
                elif first in ("N", "B"):  num_minor_pieces = max(0, num_minor_pieces - 1)
                elif first == "R":         num_rooks        = max(0, num_rooks        - 1)
                elif first == "Q":         num_queens       = max(0, num_queens       - 1)
                else:                      num_pawns        = max(0, num_pawns        - 1)

            if   prom_q: num_pawns = max(0, num_pawns - 1); num_queens       += 1
            elif prom_r: num_pawns = max(0, num_pawns - 1); num_rooks        += 1
            elif prom_m: num_pawns = max(0, num_pawns - 1); num_minor_pieces += 1

            num_pawns        = max(0, num_pawns)
            num_minor_pieces = max(0, num_minor_pieces)
            num_rooks        = max(0, num_rooks)
            num_queens       = max(0, num_queens)
            pieces_remaining = max(0, pieces_remaining)

            move_number    = (i // 2) + 1
            total_non_pawn = num_minor_pieces + num_rooks + num_queens
            is_endgame     = (
                num_queens == 0
                or pieces_remaining < 14
                or total_non_pawn < 6
            )

            if is_endgame:
                is_opening = is_middlegame = False
            else:
                is_opening = (
                    move_number < 15
                    and num_queens >= 2
                    and pieces_remaining > 26
                    and (not (wck or wcq) or not (bck or bcq))
                )
                if (wck or wcq) and (bck or bcq):        is_opening = False
                if pieces_remaining < 16 or move_number > 15: is_opening = False
                is_middlegame = not is_opening

            board_states.append({
                "pieces_remaining":        pieces_remaining,
                "material_density":        pieces_remaining / 32,
                "num_minor_pieces":        num_minor_pieces,
                "num_rooks":               num_rooks,
                "num_queens":              num_queens,
                "num_pawns":               num_pawns,
                "complexity_material_score": (
                    num_pawns + 3 * num_minor_pieces
                    + 5 * num_rooks + 9 * num_queens
                ),
                "white_castled_king":  1 if wck else 0,
                "white_castled_queen": 1 if wcq else 0,
                "black_castled_king":  1 if bck else 0,
                "black_castled_queen": 1 if bcq else 0,
                "queen_present":       1 if num_queens > 0 else 0,
                "is_opening":          1 if is_opening    else 0,
                "is_middlegame":       1 if is_middlegame else 0,
                "is_endgame":          1 if is_endgame    else 0,
            })

        return board_states

    def _classify_piece_move(self, move: str) -> dict:
        """Return one-hot piece-type flags for a SAN move string."""
        clean = move.strip("+#")
        if clean.startswith("O-O"):
            return {
                "is_pawn_move": 0, "is_knight_move": 0, "is_bishop_move": 0,
                "is_rook_move": 0, "is_queen_move":  0, "is_king_move":   1,
            }
        first = clean[0]
        return {
            "is_pawn_move":   int(first.islower()),
            "is_knight_move": int(first == "N"),
            "is_bishop_move": int(first == "B"),
            "is_rook_move":   int(first == "R"),
            "is_queen_move":  int(first == "Q"),
            "is_king_move":   int(first == "K"),
        }

    def _to_move_level(self, games: list) -> pd.DataFrame:
        """
        Convert a list of Lichess game JSON objects to a move-level DataFrame.
        """
        records = []

        for game in games:
            game_id    = game["id"]
            moves      = game.get("moves", "").split()
            clocks     = game.get("clocks", [])
            analysis   = game.get("analysis", [])
            clock_info = game.get("clock", {})
            initial    = clock_info.get("initial", 0)
            increment  = clock_info.get("increment", 0)
            time_ctrl  = f"{initial // 60}+{increment}"

            white_user = game["players"]["white"].get("user", {}).get("name", "")
            black_user = game["players"]["black"].get("user", {}).get("name", "")

            board_states = self._track_board_state(moves)
            evals_list   = []

            for i, move in enumerate(moves):
                color       = "white" if i % 2 == 0 else "black"
                move_number = (i // 2) + 1

                time_left        = clocks[i] if i < len(clocks) else None
                time_spent       = None
                time_left_ratio  = None
                time_spent_ratio = 0

                if time_left is not None:
                    time_left_ratio = (time_left / 100) / initial if initial > 0 else 0

                    if color == "white":
                        if i >= 2 and i - 2 < len(clocks):
                            time_spent = clocks[i - 2] - time_left
                        elif i == 0:
                            time_spent = (initial * 100) - time_left
                    else:
                        if i >= 2 and i - 2 < len(clocks):
                            time_spent = clocks[i - 2] - time_left
                        elif i == 1:
                            time_spent = (initial * 100) - time_left

                    if time_spent is not None and initial > 0:
                        time_spent_ratio = (time_spent / 100) / initial
                    else:
                        time_spent_ratio = 0
                else:
                    time_spent_ratio = 0

                eval_data   = analysis[i] if i < len(analysis) else {}
                chosen_eval = eval_data.get("eval")
                chosen_mate = eval_data.get("mate")

                evals_list.append(chosen_eval)

                vol = None
                if i >= 2:
                    recent = [
                        evals_list[j]
                        for j in range(max(0, i - 2), i + 1)
                        if evals_list[j] is not None
                    ]
                    if len(recent) >= 2:
                        vol = float(np.std(recent))

                bs = board_states[i] if i < len(board_states) else {}
                pf = self._classify_piece_move(move)

                records.append({
                    "game_id":         game_id,
                    "move_number":     move_number,
                    "color":           color,
                    "username":        white_user if color == "white" else black_user,
                    "move":            move,
                    "time_control":    time_ctrl,
                    "increment_sec":   increment,
                    "time_left_cs":    time_left,
                    "time_left_sec":   (time_left / 100) if time_left else None,
                    "time_spent_cs":   time_spent,
                    "time_spent_sec":  (time_spent / 100) if time_spent else None,
                    "time_left_ratio":  time_left_ratio if time_left_ratio is not None else 0.0,
                    "time_spent_ratio": time_spent_ratio,
                    "eval":            chosen_eval,
                    "mate":            chosen_mate,
                    "best_move":       eval_data.get("best"),
                    "variation":       eval_data.get("variation"),
                    "judgment":        eval_data.get("judgment", {}).get("name"),
                    "eval_volatility": vol,
                    **{k: bs.get(k) for k in [
                        "pieces_remaining", "material_density",
                        "num_minor_pieces", "num_rooks",
                        "num_queens", "num_pawns",
                        "complexity_material_score",
                        "white_castled_king", "white_castled_queen",
                        "black_castled_king", "black_castled_queen",
                        "queen_present",
                        "is_opening", "is_middlegame", "is_endgame",
                    ]},
                    **pf,
                })

        return pd.DataFrame(records)

if __name__=="__main__":
    obj=DataIngestion()
    df = obj.fetch_user_games('stevenhan', 'rapid+classical', 10000)
    base = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(base, '../../notebook/data/stevenhan_move_level.csv'), index = False )