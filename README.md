# Chess-Error-Analysis-and-Prediction

Dashboard Link: http://chess-error-analysis-prediction-5-env.eba-np5widfm.us-east-1.elasticbeanstalk.com/

### Predicting blunders and inaccuracies in chess games using player-specific machine learning models:

* Identifying the problem space and determining what chess error patterns can be predicted
* Sourcing game data from the Lichess API and determining what move-level information is extractable
* Use LLM to brainstorm potential new features
* Performing feature engineering to derive time pressure, board complexity, eval volatility, and interaction terms from raw game data
* Using domain knowledge to separate the problem into four modelling tasks: blunder and inaccuracy prediction across two time controls (blitz and standard)
* Perform hypothesis testing on my data (username 'stevenhan') to learn more about the data and what I can model
* Explore Lichess users to collect data from. Decide on user 'clownmitmuetze' for standard (rapid+classical) games due to larger quantity of analyzed rapid games closer to the user strengh I am targeting, and user 'DoktorAvalanche' for blitz due to larger quantity of analyzed blitz games closer to the user strengh I am targeting
* Perform Exploratory data analysis on each dataset to understand error distributions and feature relationships
* Perform Model training, feature selection via RFECV, and hyperparameter tuning separately for each time control and error type
* Selecting the best model per task based on PR AUC as the primary metric, given the small percentage of blunders and inaccuracies.
* Implementing the full pipeline in a modular Python architecture
* Use LLM to help build a Flask + Dash web application for descriptive analytics and personalized predictive tools
* AWS deployment


#### 1) Problem Statement
Chess players make errors such as blunders (≥300 centipawn eval loss) and inaccuracies (100–200cp) at rates influenced by time pressure, board complexity, and position volatility. This project builds personalized models trained on an individual player's game history to predict the likelihood of an error on any given move, and deploys those models in an interactive web application.

#### 2) Data Collection
Game data is fetched from the Lichess public API for a given username. Only games with existing computer analysis are used. Data is parsed to move-level records, capturing clock times, engine evaluations, board state, and move information for every move in every game.

#### 3) Feature Engineering
Move-level features are derived from the raw API data, including time pressure normalization, eval volatility (rolling standard deviation of engine eval over last 3 moves), material complexity scores, castling flags, piece move types, and several interaction terms. A unified eval column combines centipawn and forced-mate evaluations into a single continuous variable.  

* Features given in Lichess API
    - game_id
    - move_number
    - color
    - username
    - move
    - time_control
    - increment_sec
    - time_left_sec
    - time_spent_sec
    - eval
    - mate  

* Features created
    - pieces_remaining
    - material_density
    - number of each type of piece
    - complexity_material_score
    - which sides castled
    - queen_present
    - is_opening, is_middlegame, is_endgame 
    - complexity_material_score: number of pieces weighted by piece type
    - which piece move
    - avg_time_spent_per_move: average time spent per move per game
    - time_spent_ratio
    - eval_unified: combine the eval and mate column into one. The logic is that mate situations should be more extreme than any centipawn eval in the dataset, but not to infinity. Mate scores get mapped into a band that sits above the greatest eval depending on how many moves away mate is. A many move mate gets a lower eval than a mate that is close by.
    - eval_volatility: rolling std of eval_unified over last 3 moves
    - is_mate_threat, is_checkmate
    - eval_loss
    - is_inaccuracy, is_mistake, is_blunder: the target binary vars
    - color_white
    - complxity material norm: Normalized material complexity score
    - eval volatility norm: Normalized eval volatility score
    - time_pressure_norm: Normalized time pressure  

* Interaction terms:  

    - material_time_pressure_int: complexity_material_norm x time_pressure_norm: The idea is that a complicated position is only dangerous when the player is short on time. A position with many pieces, queens still on the board, and high material density requires calculation. But if the player have plenty of time, they can work through the position carefully and find the right move. It's only under time pressure when the complexity becomes a blunder risk.  
    
    - time_eval_volatility_int: time_pressure_norm x eval_volatility_norm: Sharpness of the position alone is not fatal if the player has time to calculate. Time pressure in a stable, quiet position is less dangerous. But sharp position + time pressure is exactly the scenario where players consistently blunder  

    - late_endgame_int: move_number_norm x is_endgame: is_endgame is a binary flag, but not all endgame moves are equally risky. The move_number_norm component tracks how far into the game the player is relative to that game's total length.  

    - cumulative_time_pressure: expanding mean of time_spent_ratio. By the time the player reaches move 20, the model knows whether they have been consistently rushing (high expanding mean throughout) or whether this single long move was an isolated deep think  

#### 4) Exploratory Data Analysis
EDA and hypothesis testing are performed on each dataset to examine how error rates vary across time pressure bins, game phase, board complexity, and eval volatility. This informed both feature selection and the decision to model blitz and standard time controls separately.

#### 5) Model Training
Four separate models are trained, one per combination of time control (blitz / standard) and error type (blunder / inaccuracy). Class imbalance to capture the blunder / inaccuracy signal is handled via scale_pos_weight. Models evaluated include XGBoost and CatBoost, with PR AUC as the primary metric given the rarity of errors relative to normal moves.

#### 6) Feature Selection
RFECV is applied using a CatBoost estimator with stratified cross-validation to identify the optimal feature subset for each of the four tasks. Blunder models are compressed to 7–9 features; inaccuracy models retained around 16 features, reflecting the greater difficulty of predicting subtler errors.

#### 7) Hyperparameter Tuning
RandomizedSearchCV with stratified k-fold cross-validation is used to tune each model, optimizing for PR AUC.

#### 8) Modular Implementation
The full pipeline is implemented in a modular way with exception handling and logger: components folder containing data_ingestion.py, data_transformation.py, and model_trainer.py. Pipeline folder containing predict_pipeline.py and train_pipeline.py. Application.py (Flask server and Dash dashboard).

#### 9) Flask + Dash Web Application
The web application allows a user to enter their Lichess username, load their game history, and automatically train personalized models on their data. The dashboard provides two sections of descriptive analytics (error rates by time pressure, game phase, complexity, and volatility) and predictive tools (time trouble threshold analysis, post-game error attribution, and a game risk timeline). The risk timeline accepts any Lichess game URL and plots predicted blunder and inaccuracy risk across every move alongside actual errors.  

<img width="398" height="434" alt="loadpage" src="https://github.com/user-attachments/assets/d00078fc-34c6-42c4-9f62-6a6befef31ff" />  

Login page  

<img width="923" height="463" alt="time_pressure" src="https://github.com/user-attachments/assets/ea9e2a8a-6112-4ad0-a05e-30966f2f6c3d" />    

Error rate by time remaining bins

<img width="898" height="310" alt="trend_lines" src="https://github.com/user-attachments/assets/2c0afcf3-73f9-4098-88cc-e674df6b4779" />    

Error rate vs time remaining plotted as line chart

<img width="905" height="491" alt="errorrate_gamephase" src="https://github.com/user-attachments/assets/be66d3e7-3397-40ab-8a42-3559d37acc53" />    

Error rate by game phase and summary table

<img width="911" height="506" alt="errorrate_volatility" src="https://github.com/user-attachments/assets/624fcba4-dede-431d-950d-5bfdfa3cfe18" />    

Error rate by engine evaluation volatility and summary table

<img width="907" height="511" alt="errorrate_complexity" src="https://github.com/user-attachments/assets/a5434ceb-6534-4c45-9199-df59022137a7" />    

Error rate by position complexity quartile and summary table

<img width="896" height="491" alt="timespentratio_distribution" src="https://github.com/user-attachments/assets/705530ca-a4af-45c6-8c62-841709f3164e"/>    

Time spent ratio by error category box plots and summary table

<img width="923" height="465" alt="blunder_threshold" src="https://github.com/user-attachments/assets/50a4fc4f-9a3b-44c2-9782-52ced0e3518b" />    

Blunder rate threshold predictive tool

<img width="871" height="520" alt="timeline_prediction_chart" src="https://github.com/user-attachments/assets/e4aed248-9b25-47aa-8649-d8bc3e4c1c43" />    

Game risk timeline predictive tool

<img width="897" height="502" alt="gamesummary_move_detail" src="https://github.com/user-attachments/assets/d59008ff-1216-4da8-aa76-f42dbd693e1a" />    

Sample game with move-by-move chess error prediction probability and if player made actual error 

<img width="931" height="476" alt="post_game_error" src="https://github.com/user-attachments/assets/06967621-80b8-4e39-83f9-47d8fd4a47c5" />    

Post game analysis tool for error analysis, the dominant game phase and primary error driver

#### 10) AWS Deployment
The application is deployed to AWS using Elastic Beanstalk and CodePipeline for continuous deployment.

#### Conclusion
This project produces a personalized error and analytics tool for chess players, built on their own game histories. Key challenges included engineering complex new features, capturing error signals that occur in only 1–5% of moves, developing separate models for faster and slower time controls, and distinguishing between inaccuracies and blunders. Given the rarity of errors, PR AUC was used as the primary evaluation metric. The result is a deployable, player-specific tool that translates raw game data into interpretable error risk predictions.

