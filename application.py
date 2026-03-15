"""
application.py
Main Flask + Dash web application.
Descriptive analytics and predictive tools for chess error analysis.

Run:   python application.py
Home:  http://localhost:5000/
Dash:  http://localhost:5000/dashboard/
"""

import os
import flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output, State

from src.pipeline.train_pipeline   import TrainPipeline
from src.pipeline.predict_pipeline import PredictPipeline, models_available

server = flask.Flask(__name__)
server.secret_key = 'chess_analysis_secret'

_state = {'df': None, 'username': None}

@server.route('/')
def home():
    return flask.render_template('home.html')


@server.route('/load', methods=['POST'])
def load_data():
    username  = flask.request.form.get('username', '').strip()
    perf_type = flask.request.form.get('perf_type', 'blitz')
    max_games = int(flask.request.form.get('max_games', 5000))

    if not username:
        return flask.render_template('home.html', error='Please enter a username.')

    try:
        pipeline = TrainPipeline()
        df = pipeline.run(username=username, perf_type=perf_type, max_games=max_games)

        os.makedirs('data', exist_ok=True)
        df.to_csv(f'data/{username}_processed.csv', index=False)

        _state['df']        = df
        _state['username']  = username
        _state['perf_type'] = perf_type

    except Exception as e:
        return flask.render_template('home.html', error=f'Error loading data: {str(e)}')

    return flask.redirect('/dashboard/')

@server.route('/health')
def health():
    return 'OK', 200

ERROR_COLS   = ['is_inaccuracy', 'is_mistake', 'is_blunder']
ERROR_LABELS = ['Inaccuracy', 'Mistake', 'Blunder']
COLORS       = {'Inaccuracy': '#4C72B0', 'Mistake': '#DD8452', 'Blunder': '#C44E52'}
PHASE_ORDER  = ['Opening', 'Middlegame', 'Endgame']
CAT_ORDER    = ['Normal', 'Inaccuracy', 'Mistake', 'Blunder']

TABLE_STYLE = dict(
    style_cell={'fontFamily': 'Arial, sans-serif', 'fontSize': '12px',
                'textAlign': 'center', 'padding': '6px'},
    style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0',
                  'borderBottom': '2px solid #ccc'},
    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}],
    style_table={'overflowX': 'auto'},
)

SHOW = {'display': 'block'}
HIDE = {'display': 'none'}

app = Dash(__name__, server=server, url_base_pathname='/dashboard/',
           suppress_callback_exceptions=True)
app.title = 'Chess Error Analysis'

def _card(title, *children):
    return html.Div(style={
        'backgroundColor': 'white', 'border': '1px solid #e0e0e0',
        'borderRadius': '4px', 'padding': '16px 20px', 'marginBottom': '16px',
    }, children=[
        html.H4(title, style={'margin': '0 0 12px', 'fontSize': '14px',
                              'color': '#333', 'fontWeight': 'bold'}),
        *children,
    ])


_input_style  = {'width': '100%', 'padding': '8px 10px', 'fontSize': '13px',
                 'border': '1px solid #ccc', 'borderRadius': '3px',
                 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}
_btn_style    = {'padding': '8px 20px', 'fontSize': '13px', 'background': '#333',
                 'color': 'white', 'border': 'none', 'borderRadius': '3px',
                 'cursor': 'pointer', 'marginTop': '4px'}
_label_style  = {'fontSize': '13px', 'marginBottom': '4px', 'display': 'block',
                 'fontWeight': 'bold', 'color': '#444'}

app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f7f7f7',
           'minHeight': '100vh'},
    children=[

        html.Div(style={'padding': '14px 32px', 'borderBottom': '1px solid #ddd',
                        'backgroundColor': 'white', 'display': 'flex',
                        'alignItems': 'center', 'justifyContent': 'space-between'},
            children=[
                html.Div([
                    html.H2('Chess Error Analysis',
                            style={'margin': 0, 'fontSize': '18px', 'color': '#222'}),
                    html.Div(id='header-username',
                             style={'fontSize': '12px', 'color': '#888', 'marginTop': '2px'}),
                ]),
                html.A('Load New Data', href='/',
                       style={'fontSize': '12px', 'color': '#555', 'textDecoration': 'none'}),
            ]),

        html.Div(style={'padding': '16px 32px'}, children=[

            # Top-level tabs
            dcc.Tabs(id='main-tabs', value='tab-desc',
                     style={'marginBottom': '16px'},
                     children=[
                         dcc.Tab(label='Descriptive Analytics', value='tab-desc'),
                         dcc.Tab(label='Predictive Tools',      value='tab-pred'),
                     ]),

            html.Div(id='desc-section', children=[
                dcc.Tabs(id='desc-sub-tabs', value='tab-time',
                         style={'marginBottom': '14px'},
                         children=[
                             dcc.Tab(label='Time Pressure', value='tab-time'),
                             dcc.Tab(label='Game Phase',    value='tab-phase'),
                             dcc.Tab(label='Complexity',    value='tab-complexity'),
                             dcc.Tab(label='Volatility',    value='tab-volatility'),
                             dcc.Tab(label='Distributions', value='tab-dist'),
                         ]),
                html.Div(id='desc-content'),
            ]),

            html.Div(id='pred-section', style=HIDE, children=[
                dcc.Tabs(id='pred-sub-tabs', value='tab-threshold',
                         style={'marginBottom': '14px'},
                         children=[
                             dcc.Tab(label='Time Trouble Threshold', value='tab-threshold'),
                             dcc.Tab(label='Game Risk Timeline',     value='tab-timeline'),
                             dcc.Tab(label='Post-Game Analysis',     value='tab-postgame'),
                         ]),

                # Time threshold content (rendered by callback)
                html.Div(id='threshold-content'),

                html.Div(id='timeline-panel', style=HIDE, children=[
                    _card('Game Risk Timeline',
                        html.P([
                            'Paste any Lichess game URL below. The game must have computer analysis already run.',
                            html.Br(),
                            'The chart shows three layers: ',
                            html.B('lines'), ' — predicted blunder and inaccuracy risk % across every move; ',
                            html.B('markers'), ' — actual blunders (●) and inaccuracies (●) plotted at their move number; ',
                            html.B('threshold lines'), ' — dashed horizontal lines showing where the model\'s danger zone begins. ',
                            'Models used automatically match the time control you loaded (blitz or standard).',
                        ], style={'fontSize': '13px', 'color': '#555', 'marginBottom': '14px', 'lineHeight': '1.7'}),
                        html.Label('Lichess Game URL', style=_label_style),
                        dcc.Input(id='timeline-url-input', type='text',
                                  placeholder='https://lichess.org/abc12345',
                                  debounce=False, style=_input_style),
                        html.Label('Analyze as', style=_label_style),
                        dcc.RadioItems(
                            id='timeline-color-select',
                            options=[
                                {'label': 'White', 'value': 'white'},
                                {'label': 'Black', 'value': 'black'},
                            ],
                            value='white',
                            inline=True,
                            style={'fontSize': '13px', 'marginBottom': '8px', 'gap': '16px'},
                        ),
                        html.Button('Analyze Game', id='timeline-btn', n_clicks=0, style=_btn_style),
                        html.Div(id='timeline-output', style={'marginTop': '16px'}),
                    ),
                ]),

                html.Div(id='postgame-panel', style=HIDE, children=[
                    _card('Post-Game Error Analysis',
                        html.P('Enter a Lichess game URL from your loaded games to see '
                               'a breakdown of what drove errors in that game.',
                               style={'fontSize': '13px', 'color': '#555', 'marginBottom': '14px'}),
                        html.Label('Lichess Game URL or Game ID', style=_label_style),
                        dcc.Input(id='game-url-input', type='text',
                                  placeholder='https://lichess.org/abc12345  or  abc12345',
                                  debounce=False, style=_input_style),
                        html.Button('Analyze Game', id='analyze-btn', n_clicks=0, style=_btn_style),
                        html.Div(id='postgame-output', style={'marginTop': '16px'}),
                    ),
                ]),
            ]),
        ]),
    ],
)


def _no_data():
    return html.P('No data loaded. Please return to the home page and load games.',
                  style={'color': '#888', 'fontSize': '13px'})


def _grouped_bar(data, group_col, group_order=None, title='', xlab=''):
    rates = data.groupby(group_col, observed=True)[ERROR_COLS].mean().reset_index()
    rates[ERROR_COLS] = rates[ERROR_COLS] * 100
    fig = go.Figure()
    for col, label in zip(ERROR_COLS, ERROR_LABELS):
        fig.add_trace(go.Bar(
            x=rates[group_col].astype(str), y=rates[col].round(2),
            name=label, marker_color=COLORS[label],
        ))
    fig.update_layout(
        barmode='group', title=title, xaxis_title=xlab,
        yaxis_title='Error Rate (%)', plot_bgcolor='white',
        paper_bgcolor='white', height=380,
        margin=dict(t=40, b=60, l=50, r=20), legend_title='Error Type',
    )
    if group_order:
        fig.update_xaxes(categoryorder='array',
                         categoryarray=[str(g) for g in group_order])
    return fig


def _summary_table(data, group_col, group_order=None):
    groups = (group_order if group_order
              else sorted(data[group_col].dropna().unique().tolist(), key=str))
    rows = []
    for g in groups:
        subset = data[data[group_col].astype(str) == str(g)]
        n = len(subset)
        row = {'Group': str(g), 'Total Moves': n}
        for col, label in zip(ERROR_COLS, ERROR_LABELS):
            row[f'{label} Count'] = int(subset[col].sum())
            row[f'{label} Rate %'] = round(float(subset[col].mean()) * 100, 2) if n > 0 else 0.0
        rows.append(row)
    tdf = pd.DataFrame(rows)
    return dash_table.DataTable(
        data=tdf.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in tdf.columns],
        **TABLE_STYLE,
    )


def _prep_df(df):
    """Add display columns to df if not already present."""
    df = df.copy()
    if 'game_phase' not in df.columns:
        df['game_phase'] = 'Unknown'
        df.loc[df['is_opening']    == 1, 'game_phase'] = 'Opening'
        df.loc[df['is_middlegame'] == 1, 'game_phase'] = 'Middlegame'
        df.loc[df['is_endgame']    == 1, 'game_phase'] = 'Endgame'
    if 'error_category' not in df.columns:
        df['error_category'] = 'Normal'
        df.loc[df['is_inaccuracy'] == 1, 'error_category'] = 'Inaccuracy'
        df.loc[df['is_mistake']    == 1, 'error_category'] = 'Mistake'
        df.loc[df['is_blunder']    == 1, 'error_category'] = 'Blunder'
    if 'time_left_bin' not in df.columns:
        df['time_left_bin'] = pd.cut(
            df['time_left_ratio_clipped'], bins=10,
            labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)]
        )
    if 'complexity_quartile' not in df.columns and 'complexity_material_norm' in df.columns:
        df['complexity_quartile'] = pd.qcut(
            df['complexity_material_norm'], q=4,
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop'
        )
    if 'volatility_quartile' not in df.columns and 'eval_volatility' in df.columns:
        valid = df['eval_volatility'].dropna()
        df.loc[valid.index, 'volatility_quartile'] = pd.qcut(
            valid, q=4,
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop'
        )
    return df


def _render_time(df):
    bin_order = [f'{i*10}-{(i+1)*10}%' for i in range(10)]
    fig_bar  = _grouped_bar(df, 'time_left_bin', bin_order,
                            'Error Rate by Time Remaining',
                            'Time Left (0% = almost no time, 100% = full time)')
    rates    = df.groupby('time_left_bin', observed=True)[ERROR_COLS].mean() * 100
    fig_line = go.Figure()
    for col, label in zip(ERROR_COLS, ERROR_LABELS):
        fig_line.add_trace(go.Scatter(
            x=[str(b) for b in rates.index], y=rates[col].round(2).tolist(),
            mode='lines+markers', name=label,
            line=dict(color=COLORS[label], width=2), marker=dict(size=6),
        ))
    fig_line.update_layout(
        title='Error Rate Trend vs Time Remaining',
        xaxis_title='Time Left Bin', yaxis_title='Error Rate (%)',
        plot_bgcolor='white', paper_bgcolor='white',
        height=340, margin=dict(t=40, b=80, l=50, r=20),
    )
    fig_line.update_xaxes(categoryorder='array', categoryarray=bin_order)
    return html.Div([
        _card('Error Rate by Time Pressure', dcc.Graph(figure=fig_bar)),
        _card('Trend Lines',                 dcc.Graph(figure=fig_line)),
        _card('Summary Table', _summary_table(df, 'time_left_bin', group_order=bin_order)),
    ])


def _render_phase(df):
    fig = _grouped_bar(df, 'game_phase', PHASE_ORDER,
                       'Error Rate by Game Phase', 'Game Phase')
    return html.Div([
        _card('Error Rate by Game Phase', dcc.Graph(figure=fig)),
        _card('Summary Table', _summary_table(df, 'game_phase', group_order=PHASE_ORDER)),
    ])


def _render_complexity(df):
    if 'complexity_quartile' not in df.columns:
        return html.P('complexity_material_norm not available.')
    q_order = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    fig = _grouped_bar(df, 'complexity_quartile', q_order,
                       'Error Rate by Complexity Quartile',
                       'Material Complexity (weighted piece score)')
    return html.Div([
        _card('Error Rate by Complexity', dcc.Graph(figure=fig)),
        _card('Summary Table', _summary_table(df, 'complexity_quartile', group_order=q_order)),
    ])


def _render_volatility(df):
    if 'volatility_quartile' not in df.columns:
        return html.P('eval_volatility not available.')
    q_order = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    dv = df.dropna(subset=['volatility_quartile']).copy()
    fig = _grouped_bar(dv, 'volatility_quartile', q_order,
                       'Error Rate by Eval Volatility Quartile',
                       'Eval Volatility (rolling std)')
    return html.Div([
        _card('Error Rate by Volatility', dcc.Graph(figure=fig)),
        _card('Summary Table', _summary_table(dv, 'volatility_quartile', group_order=q_order)),
    ])


def _render_distributions(df):
    fig_box = px.box(
        df, x='error_category', y='time_spent_ratio',
        category_orders={'error_category': CAT_ORDER},
        title='Time Spent Ratio by Error Category',
        labels={'error_category': 'Error Type', 'time_spent_ratio': 'Time Spent Ratio'},
    )
    fig_box.update_layout(showlegend=False, plot_bgcolor='white',
                          paper_bgcolor='white', height=360,
                          margin=dict(t=40, b=40, l=50, r=20))

    rows = []
    for cat in CAT_ORDER:
        sub = df[df['error_category'] == cat]['time_spent_ratio'].dropna()
        if len(sub) == 0: continue
        rows.append({'Category': cat, 'N': len(sub),
                     'Mean':   round(float(sub.mean()), 4),
                     'Median': round(float(sub.median()), 4),
                     'Std':    round(float(sub.std()), 4)})
    tdf = pd.DataFrame(rows)
    return html.Div([
        _card('Time Spent Ratio by Error Category (Defined as the amount of time the player spent on that move divided by the average time a player spent on a move across all games)', dcc.Graph(figure=fig_box)),
        _card('Descriptive Statistics', dash_table.DataTable(
            data=tdf.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in tdf.columns],
            **TABLE_STYLE,
        )),
    ])


def _render_threshold(df):
    username  = _state.get('username')
    perf_type = _state.get('perf_type', 'blitz')
    predictor = PredictPipeline(username=username, perf_type=perf_type)
    analysis  = predictor.time_threshold_analysis(df)

    figs = []
    for target, col_label in [('blunder', 'Blunder'), ('inaccuracy', 'Inaccuracy')]:
        if target not in analysis:
            continue
        res      = analysis[target]
        rates    = res['rates']
        baseline = res['baseline']
        thresh   = res['threshold_pct']

        x_vals = [str(b) for b in rates.index]
        y_vals = rates.round(3).tolist()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_vals, y=y_vals,
                             name=f'{col_label} Rate',
                             marker_color=COLORS.get(col_label, '#555')))
        fig.add_hline(y=baseline * 2, line_dash='dash', line_color='red',
                      annotation_text=f'2x baseline ({baseline*2:.2f}%)',
                      annotation_position='top right')

        title_note = (
            f"Threshold: when {thresh}% of time is used, rate exceeds 2x player baseline (across all games)"
            if thresh else "Threshold not reached in your data"
        )
        fig.update_layout(
            title=f'{col_label} Rate vs % of Time Used<br><sup>{title_note}</sup>',
            xaxis_title='% of Time Used',
            yaxis_title='Error Rate (%)',
            plot_bgcolor='white', paper_bgcolor='white',
            height=340, margin=dict(t=60, b=60, l=50, r=20),
            showlegend=False,
        )
        figs.append((col_label, fig, res))

    if not figs:
        return html.P('Could not compute threshold — check that eval data is available.')

    cards = []
    for col_label, fig, res in figs:
        thresh = res['threshold_pct']
        note   = (
            f"Your {col_label.lower()} rate first doubles its baseline "
            f"({res['baseline']:.2f}%) once you have used {thresh}% of your time."
            if thresh else
            f"Your {col_label.lower()} rate does not clearly double within the data range."
        )
        cards.append(_card(
            f'{col_label} — Time Trouble Threshold',
            dcc.Graph(figure=fig),
            html.P(note, style={'fontSize': '13px', 'color': '#444', 'marginTop': '8px'}),
        ))

    return html.Div(cards)


def _danger_result(results):
    """Render the output of predict_danger()."""
    rows = []
    for target, label in [('blunder', 'Blunder'), ('inaccuracy', 'Inaccuracy')]:
        r    = results[target]
        prob = r['probability']
        zone = r['zone']
        rows.append(html.Tr([
            html.Td(label, style={'padding': '8px 12px', 'fontWeight': 'bold'}),
            html.Td(f'{prob}%', style={'padding': '8px 12px'}),
            html.Td(zone, style={'padding': '8px 12px'}),
        ]))

    table = html.Table(
        [html.Thead(html.Tr([html.Th('Error Type', style={'padding': '8px 12px', 'textAlign': 'left'}),
                             html.Th('Probability', style={'padding': '8px 12px', 'textAlign': 'left'}),
                             html.Th('Zone', style={'padding': '8px 12px', 'textAlign': 'left'})]))]
        + [html.Tbody(rows)],
        style={'borderCollapse': 'collapse', 'border': '1px solid #ddd',
               'width': '100%', 'fontSize': '13px'}
    )
    note = html.P(results.get('imputed_note', ''),
                  style={'fontSize': '11px', 'color': '#999', 'marginTop': '10px'})
    return html.Div([table, note])


def _postgame_result(analysis):
    """Render post-game attribution results."""
    if analysis is None:
        return html.P('Game not found in loaded data. Make sure the game is from your loaded dataset.',
                      style={'color': '#c44', 'fontSize': '13px'})

    sections = [
        html.P(f"Game: {analysis['game_id']}  |  Total moves: {analysis['total_moves']}",
               style={'fontSize': '13px', 'marginBottom': '14px', 'color': '#555'}),
    ]

    for target, label in [('blunder', 'Blunder'), ('inaccuracy', 'Inaccuracy')]:
        if target not in analysis:
            continue
        res = analysis[target]
        if res.get('count', 0) == 0:
            sections.append(html.Div([
                html.H4(f'{label} Analysis', style={'fontSize': '14px', 'margin': '0 0 6px'}),
                html.P(f'No {label.lower()}s in this game.',
                       style={'fontSize': '13px', 'color': '#888', 'marginBottom': '14px'}),
            ]))
            continue

        phase_str = ', '.join(f"{k}: {v}" for k, v in res['phase_counts'].items() if v > 0) or '—'
        feat_rows = []
        for feat, vals in res['feat_comparison'].items():
            direction = 'Higher' if vals['higher'] else 'Lower'
            feat_rows.append(html.Tr([
                html.Td(feat,                    style={'padding': '5px 10px'}),
                html.Td(str(vals['at_error']),   style={'padding': '5px 10px'}),
                html.Td(str(vals['at_normal']),  style={'padding': '5px 10px'}),
                html.Td(direction, style={'padding': '5px 10px'}),
            ]))

        feat_table = html.Table(
            [html.Thead(html.Tr([
                html.Th('Feature',       style={'padding': '5px 10px', 'textAlign': 'left'}),
                html.Th(f'Avg at {label}',   style={'padding': '5px 10px', 'textAlign': 'left'}),
                html.Th('Avg at Normal',     style={'padding': '5px 10px', 'textAlign': 'left'}),
                html.Th('Error Compared to Normal',     style={'padding': '5px 10px', 'textAlign': 'left'}),
            ]))]
            + [html.Tbody(feat_rows)],
            style={'borderCollapse': 'collapse', 'border': '1px solid #ddd',
                   'width': '100%', 'fontSize': '12px', 'marginTop': '8px'}
        )

        sections.append(html.Div(style={'marginBottom': '20px'}, children=[
            html.H4(f'{label} Analysis ({res["count"]} {label.lower()}s)',
                    style={'fontSize': '14px', 'margin': '0 0 8px'}),
            html.P([html.B('Average move: '), str(res['avg_move']),
                    '  |  ',
                    html.B('Dominant phase: '), res['dominant_phase'],
                    '  |  ',
                    html.B('Phase breakdown: '), phase_str],
                   style={'fontSize': '13px', 'marginBottom': '6px'}),
            html.P([html.B('Avg time pressure at error: '),
                    f"{res['avg_pressure_pct']}% of time used"],
                   style={'fontSize': '13px', 'marginBottom': '6px'}),
            html.P([html.B('Primary driver: '), res['primary_driver']],
                   style={'fontSize': '13px', 'marginBottom': '6px'}),
            feat_table,
        ]))

    return html.Div(sections)



# 1. Header username
@app.callback(Output('header-username', 'children'),
              Input('main-tabs', 'value'))
def update_header(_):
    u = _state.get('username')
    return f'Player: {u}' if u else 'No data loaded'


# 2. Toggle main sections
@app.callback(
    [Output('desc-section', 'style'), Output('pred-section', 'style')],
    Input('main-tabs', 'value'),
)
def toggle_main(tab):
    return (SHOW, HIDE) if tab == 'tab-desc' else (HIDE, SHOW)


# 3. Descriptive sub-tab content
@app.callback(
    Output('desc-content', 'children'),
    Input('desc-sub-tabs', 'value'),
)
def render_desc(sub_tab):
    df = _state.get('df')
    if df is None:
        return _no_data()
    df = _prep_df(df)
    if sub_tab == 'tab-time':        return _render_time(df)
    if sub_tab == 'tab-phase':       return _render_phase(df)
    if sub_tab == 'tab-complexity':  return _render_complexity(df)
    if sub_tab == 'tab-volatility':  return _render_volatility(df)
    if sub_tab == 'tab-dist':        return _render_distributions(df)
    return html.Div()


# 4. Predictive sub-tab: show threshold content + toggle panels
@app.callback(
    [Output('threshold-content', 'children'),
     Output('timeline-panel',    'style'),
     Output('postgame-panel',    'style')],
    Input('pred-sub-tabs', 'value'),
)
def render_pred(sub_tab):
    df = _state.get('df')

    if sub_tab == 'tab-threshold':
        if df is None:
            return _no_data(), HIDE, HIDE
        return _render_threshold(df), HIDE, HIDE

    if sub_tab == 'tab-timeline':
        return html.Div(), SHOW, HIDE

    if sub_tab == 'tab-postgame':
        return html.Div(), HIDE, SHOW

    return html.Div(), HIDE, HIDE


# 5. Game Risk Timeline
@app.callback(
    Output('timeline-output', 'children'),
    Input('timeline-btn', 'n_clicks'),
    State('timeline-url-input', 'value'),
    State('timeline-color-select', 'value'),
    prevent_initial_call=True,
)
def compute_timeline(n_clicks, game_url, color):
    if not game_url or not game_url.strip():
        return html.P('Please enter a Lichess game URL.',
                      style={'color': '#c44', 'fontSize': '13px'})

    username  = _state.get('username')
    perf_type = _state.get('perf_type', 'blitz')

    if not username:
        return _no_data()

    # Extract 8-character game ID from URL or raw ID
    raw     = game_url.strip().split('#')[0].rstrip('/')
    game_id = raw.split('lichess.org/')[-1][:8] if 'lichess.org/' in raw else raw[:8]

    try:
        if not models_available(username, perf_type):
            return html.P('Models not found — please reload your data first.',
                          style={'color': '#c44', 'fontSize': '13px'})

        predictor = PredictPipeline(username=username, perf_type=perf_type)
        df_game   = predictor.game_timeline(game_id)
        df_pred   = df_game[df_game['color'] == color].copy()
        return _render_timeline(df_pred, game_id, username, perf_type, color)

    except Exception as e:
        return html.P(f'Error: {str(e)}',
                      style={'color': '#c44', 'fontSize': '13px'})


def _render_timeline(df, game_id, username, perf_type, color):
    """Build risk timeline chart + summary table + move-level detail."""
    df = df.sort_values('move_number').reset_index(drop=True)
    tc_lbl = 'Blitz' if perf_type == 'blitz' else 'Standard'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['move_number'], y=(df['blunder_prob'] * 100).round(3),
        mode='lines', name='Blunder Risk %',
        line=dict(color='#C44E52', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df['move_number'], y=(df['inaccuracy_prob'] * 100).round(3),
        mode='lines', name='Inaccuracy Risk %',
        line=dict(color='#4C72B0', width=2),
    ))

    # Actual blunders — red circle markers
    blunders = df[df['is_blunder'] == 1]
    if len(blunders) > 0:
        fig.add_trace(go.Scatter(
            x=blunders['move_number'],
            y=(blunders['blunder_prob'] * 100).round(3),
            mode='markers', name='Actual Blunder (≥300cp)',
            marker=dict(color='#C44E52', size=8, symbol='circle',
                        line=dict(width=2, color='#C44E52')),
        ))

    # Actual inaccuracies — blue circle markers
    inaccuracies = df[df['is_inaccuracy'] == 1]
    if len(inaccuracies) > 0:
        fig.add_trace(go.Scatter(
            x=inaccuracies['move_number'],
            y=(inaccuracies['inaccuracy_prob'] * 100).round(3),
            mode='markers', name='Actual Inaccuracy (≥100cp)',
            marker=dict(color='#4C72B0', size=8, symbol='circle',
                        line=dict(width=2, color='#4C72B0')),
        ))

    fig.update_layout(
        title=f'Risk Timeline — {game_id}  ({tc_lbl} model, {color.capitalize()})',
        xaxis_title='Move Number',
        yaxis_title='Predicted Risk (%)',
        plot_bgcolor='white', paper_bgcolor='white',
        height=460, margin=dict(t=50, b=70, l=50, r=20),
        legend=dict(orientation='h', y=-0.2),
        xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=False),
        yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=False),
    )

    total        = len(df)
    n_blunder    = int(df['is_blunder'].sum())
    n_inaccuracy = int(df['is_inaccuracy'].sum())
    danger_b     = int((df['blunder_zone']    == 'Danger').sum())
    danger_i     = int((df['inaccuracy_zone'] == 'Danger').sum())
    caught_b     = int(((df['blunder_zone']    == 'Danger') & (df['is_blunder']    == 1)).sum())
    caught_i     = int(((df['inaccuracy_zone'] == 'Danger') & (df['is_inaccuracy'] == 1)).sum())
    misclass_b   = int(((df['blunder_zone']    == 'Safe')   & (df['is_blunder']    == 1)).sum())
    misclass_i   = int(((df['inaccuracy_zone'] == 'Safe')   & (df['is_inaccuracy'] == 1)).sum())

    summary_rows = [
        {'Metric': 'Actual errors in game',          'Blunder': n_blunder,    'Inaccuracy': n_inaccuracy},
        {'Metric': 'Moves flagged as Danger zone',   'Blunder': danger_b,     'Inaccuracy': danger_i},
        {'Metric': 'Actual errors caught in Danger', 'Blunder': caught_b,     'Inaccuracy': caught_i},
        {'Metric': 'Misclassifications (Safe but error occured)',
         'Blunder': misclass_b, 'Inaccuracy': misclass_i},
        {'Metric': 'Misclassification % (of all moves)',
         'Blunder':    round(misclass_b / total * 100, 1),
         'Inaccuracy': round(misclass_i / total * 100, 1)},
        {'Metric': 'Avg predicted move risk %',
         'Blunder':    round(float(df['blunder_prob'].mean()    * 100), 3),
         'Inaccuracy': round(float(df['inaccuracy_prob'].mean() * 100), 3)},
        {'Metric': 'Peak predicted move risk %',
         'Blunder':    round(float(df['blunder_prob'].max()    * 100), 3),
         'Inaccuracy': round(float(df['inaccuracy_prob'].max() * 100), 3)},
    ]
    summary_table = dash_table.DataTable(
        data=summary_rows,
        columns=[{'name': c, 'id': c} for c in ['Metric', 'Blunder', 'Inaccuracy']],
        **TABLE_STYLE,
    )

    detail_cols = ['move_number', 'color', 'move',
                   'blunder_prob', 'inaccuracy_prob',
                   'blunder_zone', 'inaccuracy_zone',
                   'is_blunder', 'is_inaccuracy']
    detail_cols = [c for c in detail_cols if c in df.columns]
    detail = df[detail_cols].copy()

    if 'blunder_prob'    in detail.columns:
        detail['blunder_prob']    = (detail['blunder_prob']    * 100).round(3)
    if 'inaccuracy_prob' in detail.columns:
        detail['inaccuracy_prob'] = (detail['inaccuracy_prob'] * 100).round(3)

    rename_map = {
        'move_number':    'Move',      'color':          'Color',
        'move':           'SAN',       'blunder_prob':   'Blunder Risk %',
        'inaccuracy_prob':'Inaccuracy Risk %',
        'blunder_zone':   'Blunder Zone',
        'inaccuracy_zone':'Inaccuracy Zone',
        'is_blunder':     'Actual Blunder',
        'is_inaccuracy':  'Actual Inaccuracy',
    }
    detail = detail.rename(columns={k: v for k, v in rename_map.items() if k in detail.columns})

    zone_description = html.Div([
        html.P([
            html.B('Zone definitions: '),
            html.Span('Safe', style={'color': '#2e7d32', 'fontWeight': 'bold'}),
            ': below the 75th percentile of predicted risk in this game.  ',
            html.Span('Caution', style={'color': '#b8860b', 'fontWeight': 'bold'}),
            ': 75th to 90th percentile of predicted risk in this game.  ',
            html.Span('Danger', style={'color': '#c62828', 'fontWeight': 'bold'}),
            ': above the 90th percentile of predicted risk in this game.',
        ], style={'fontSize': '12px', 'color': '#555', 'marginBottom': '10px',
                'padding': '8px 12px', 'backgroundColor': '#f9f9f9',
                'border': '1px solid #e0e0e0', 'borderRadius': '3px',
                'lineHeight': '1.8'}),
    ], style={'marginBottom': '8px'})

    detail_table = dash_table.DataTable(
        data=detail.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in detail.columns],
        page_size=20,
        style_cell={'fontFamily': 'Arial, sans-serif', 'fontSize': '12px',
                    'textAlign': 'center', 'padding': '6px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0',
                    'borderBottom': '2px solid #ccc'},
        style_table={'overflowX': 'auto'},
        style_data_conditional=[

            {'if': {'filter_query': '{Blunder Zone} = Safe',    'column_id': 'Blunder Zone'},
             'color': '#2e7d32'},
            {'if': {'filter_query': '{Blunder Zone} = Caution', 'column_id': 'Blunder Zone'},
             'color': '#b8860b'},
            {'if': {'filter_query': '{Blunder Zone} = Danger',  'column_id': 'Blunder Zone'},
             'color': '#c62828'},
            {'if': {'filter_query': '{Inaccuracy Zone} = Safe',    'column_id': 'Inaccuracy Zone'},
             'color': '#2e7d32'},
            {'if': {'filter_query': '{Inaccuracy Zone} = Caution', 'column_id': 'Inaccuracy Zone'},
             'color': '#b8860b'},
            {'if': {'filter_query': '{Inaccuracy Zone} = Danger',  'column_id': 'Inaccuracy Zone'},
             'color': '#c62828'},

            # (Move, Color, SAN, Blunder Risk %, Blunder Zone, Actual Blunder)
            *[
                {'if': {'filter_query': f'{{Blunder Zone}} = {zone} && {{Actual Blunder}} = 1',
                        'column_id': col},
                 'fontWeight': 'bold'}
                for zone in ('Caution', 'Danger')
                for col in ('Move', 'Color', 'SAN', 'Blunder Risk %', 'Blunder Zone', 'Actual Blunder')
            ],

            # (Move, Color, SAN, Inaccuracy Risk %, Inaccuracy Zone, Actual Inaccuracy)
            *[
                {'if': {'filter_query': f'{{Inaccuracy Zone}} = {zone} && {{Actual Inaccuracy}} = 1',
                        'column_id': col},
                 'fontWeight': 'bold'}
                for zone in ('Caution', 'Danger')
                for col in ('Move', 'Color', 'SAN', 'Inaccuracy Risk %', 'Inaccuracy Zone', 'Actual Inaccuracy')
            ],
        ],
    )

    return html.Div([
        _card(f'Risk Timeline — {game_id}', dcc.Graph(figure=fig)),
        _card('Game Summary', summary_table),
        _card('Move-by-Move Detail', html.Div([zone_description, detail_table])),
    ])


# 6. Post-game analysis
@app.callback(
    Output('postgame-output', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State('game-url-input', 'value'),
    prevent_initial_call=True,
)
def compute_postgame(n_clicks, game_url):
    if not game_url or not game_url.strip():
        return html.P('Please enter a game URL or ID.', style={'color': '#c44', 'fontSize': '13px'})

    df = _state.get('df')
    if df is None:
        return _no_data()

    raw = game_url.strip()
    if 'lichess.org/' in raw:
        after_domain = raw.split('lichess.org/')[-1]
        game_id = after_domain[:8]
    else:
        game_id = raw[:8]

    try:
        username  = _state.get('username')
        perf_type = _state.get('perf_type', 'blitz')
        predictor = PredictPipeline(username=username, perf_type=perf_type)
        analysis  = predictor.post_game_analysis(df, game_id)
        return _postgame_result(analysis)
    except Exception as e:
        return html.P(f'Error: {str(e)}', style={'color': '#c44', 'fontSize': '13px'})


if __name__ == '__main__':
    server.run(debug=True, port=5000)
