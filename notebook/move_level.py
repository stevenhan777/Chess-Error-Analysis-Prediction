import os
import sys
import json
import requests
import numpy as np
import pandas as pd

def _engineer_features(df, username):

    # Filter to only have my games:
    df = df[df['username'] == username]

    # Fix first-move time_spent 
    df.loc[df['move_number'] == 1, ['time_spent_cs', 'time_spent_sec']] = 0

    # Add increment back to time_spent
    mask = df['move_number'] != 1
    df.loc[mask, 'time_spent_sec'] = df.loc[mask, 'time_spent_sec'] + df.loc[mask, 'increment_sec']
    df.loc[mask, 'time_spent_cs']  = df.loc[mask, 'time_spent_cs']  + df.loc[mask, 'increment_sec'] * 100

    # set these remaining to zero
    df.loc[df['time_spent_cs'] < 0, 'time_spent_cs'] = 0
    df.loc[df['time_spent_sec'] < 0, 'time_spent_sec'] = 0

    df = df.dropna(subset=['time_left_cs'])

    # Add average time spent per move per game
    df['avg_time_spent_per_move'] = df.groupby(['game_id', 'username'])['time_spent_sec'].transform('mean')

    # drop the original time_spent_ratio column
    df = df.drop('time_spent_ratio', axis=1)
    # Recalcualted the new column: Calculate how much longer/shorter this move was compared to average
    df['time_spent_ratio'] = df['time_spent_sec'] / df['avg_time_spent_per_move']

    # Cap time_spent_sec to time_left_sec where violated
    mask = df['time_spent_sec'] > df['time_left_sec']
    df.loc[mask, 'time_spent_sec'] = df.loc[mask, 'time_left_sec']

    # Recalculate time_spent_ratio after capping
    df.loc[mask, 'time_spent_ratio'] = (
        df.loc[mask, 'time_spent_sec'] / df.loc[mask, 'time_left_sec'].replace(0, np.nan)
    ).fillna(0)

    def process_game_evals(group):
        group = group.sort_values('move_number').copy()
        
        group['is_mate_threat'] = ((group['mate'].notna()) & (group['mate'] != 0)).astype(int)
        group['is_checkmate'] = (group['eval'].isna() & group['mate'].isna()).astype(int)
        
        group['prev_eval'] = group['eval'].shift(1)
        group['eval_change'] = group['eval'] - group['prev_eval']
        
        eval_loss = []
        for idx, row in group.iterrows():
            if pd.isna(row['eval_change']):
                eval_loss.append(None)
            else:
                loss = -row['eval_change'] if row['color'] == 'white' else row['eval_change']
                eval_loss.append(max(0, loss))
        
        group['eval_loss'] = eval_loss
        return group

    results = []
    for game_id, group in df.groupby('game_id'):
        results.append(process_game_evals(group))

    df = pd.concat(results).reset_index(drop=True)

    # Error flags from judgment 
    df['is_inaccuracy'] = ((df['eval_loss'] >= 100) & (df['eval_loss'] < 200)).astype(int)
    df['is_mistake'] = ((df['eval_loss'] >= 200) & (df['eval_loss'] < 300)).astype(int)
    df['is_blunder'] = (df['eval_loss'] >= 300).astype(int)

    # apply a unified eval column to combine mate and eval info into one column
    max_eval = df['eval'].max()
    min_eval = df['eval'].min()
    max_white_mate = df['mate'].max()   # most positive (e.g. 31)
    max_black_mate = df['mate'].min()   # most negative (e.g. -26)

    FLOOR_BUFFER = 300
    WHITE_CAP_BUFFER = max_eval * 0.10
    BLACK_CAP_BUFFER = abs(min_eval) * 0.10

    WHITE_CAP   = max_eval + WHITE_CAP_BUFFER
    WHITE_FLOOR = max_eval + FLOOR_BUFFER
    BLACK_CAP   = min_eval - BLACK_CAP_BUFFER
    BLACK_FLOOR = min_eval - FLOOR_BUFFER

    def unified_eval(row):
        if row['is_checkmate']:
            return WHITE_CAP if row['color'] == 'white' else BLACK_CAP

        elif row['is_mate_threat']:
            m = row['mate']
            if m > 0:  # white has forced mate
                proxy = WHITE_CAP - (m - 1) * (WHITE_CAP - WHITE_FLOOR) / (max_white_mate - 1)
            else:      # black has forced mate
                proxy = BLACK_CAP + (abs(m) - 1) * (abs(BLACK_CAP) - abs(BLACK_FLOOR)) / (abs(max_black_mate) - 1)
            return proxy

        else:
            return row['eval']

    df['eval_unified'] = df.apply(unified_eval, axis=1)

    mask = df['time_spent_sec'].isnull()
    df.loc[mask, 'time_spent_sec'] = df.loc[mask, 'time_spent_cs'] / 100

    # Replace the blank values with 0
    df['time_spent_ratio'] = df['time_spent_ratio'].fillna(0)

    # Recalculate eval_volatility on eval_unified
    df = df.sort_values(['game_id', 'move_number']).reset_index(drop=True)
    vol_results = []
    for _, group in df.groupby('game_id'):
        evals = group['eval_unified'].tolist()
        vols  = [None] * len(evals)
        for i in range(len(evals)):
            if i >= 2:
                recent = [evals[j] for j in range(max(0, i-2), i+1) if evals[j] is not None]
                if len(recent) >= 2:
                    vols[i] = float(np.std(recent))
        vol_results.append(pd.Series(vols, index=group.index))
    df['eval_volatility'] = pd.concat(vol_results).sort_index()

    #  Normalised features
    df['complexity_material_norm'] = (
        df['complexity_material_score'] / df['complexity_material_score'].quantile(0.95)
    ).clip(0, 1)

    max_vol = df['eval_volatility'].quantile(0.95)
    df['eval_volatility_norm'] = (df['eval_volatility'].fillna(0) / max_vol).clip(0, 1) if max_vol > 0 else 0.0

    df['time_pressure_norm'] = (1 - df['time_left_ratio']).clip(0, 1)

    # Interaction terms
    df['material_time_pressure_int'] = df['complexity_material_norm'] * df['time_pressure_norm']
    df['time_eval_volatility_int']   = df['time_pressure_norm']        * df['eval_volatility_norm']

    df['move_number_norm'] = df.groupby('game_id')['move_number'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1)
    )
    df['late_endgame_int'] = df['move_number_norm'] * df['is_endgame']

    # Accumulated time pressure: are you consistently low on time, or did you just spend a lot on this one move?
    df['cumulative_time_pressure'] = df.groupby(['game_id', 'color'])['time_spent_ratio'].transform(
        lambda x: x.expanding().mean())

    df = df.drop([ 'username', 'move', 'time_left_cs', 'time_spent_cs', 'best_move',
       'variation', 'judgment', 'prev_eval',
       'eval_change', 'eval_loss', 'time_control'], axis = 1) # Note the time_control col dropped
    
    df = pd.get_dummies(df, columns=['color'],drop_first=True, dtype=int)

    return df

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, 'data/DoktorAvalanche_blitz_loaded.csv'))
    username = 'DoktorAvalanche'
    df = _engineer_features(df, username)
    df.to_csv(os.path.join(base,'data/DoktorAvalanche_with_features_engineered.csv'), index = False)

















