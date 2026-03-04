# Chess-Error-Analysis-and-Prediction
### Predicting blunders and inaccuracies in chess games using player-specific machine learning models:

* Identifying the problem space and determining what chess error patterns are worth predicting
* Sourcing game data from the Lichess API and determining what move-level information is extractable
* Use LLM to brainstorm potential new features
* Performing feature engineering to derive time pressure, board complexity, eval volatility, and interaction terms from raw game data
* Using domain knowledge to separate the problem into four modelling tasks: blunder and inaccuracy prediction across two time controls (blitz and standard)
* Perform hypothesis testing on my data (username stevenhan) to learn more about the data and what I can model
* Explore users to collect data from, decide on user clownmitmuetze for standard (rapid+classical) due to larger quantity of analyzed rapid games closer to the user strengh I am targeting, and user DoktorAvalanche for blitz due to larger quantity of analyzed rapid games closer to the user strengh I am targeting
* Exploratory data analysis on each dataset to understand error distributions and feature relationships
* Perform Model training, feature selection via RFECV, and hyperparameter tuning separately for each time control and error type
* Selecting the best model per task based on PR AUC as the primary metric, given small percentage of blunders and inaccuracies.
* Implementing the full pipeline in a modular Python architecture
* Use LLM to help build a Flask + Dash web application for descriptive analytics and personalised predictive tools
* AWS deployment


#### 1) Problem Statement
Chess players make errors such as blunders (≥300 centipawn eval loss) and inaccuracies (100–200cp) at rates influenced by time pressure, board complexity, and position volatility. This project builds personalised models trained on an individual player's game history to predict the likelihood of an error on any given move, and deploys those models in an interactive web application.

#### 2) Data Collection
Game data is fetched from the Lichess public API for a given username. Only games with existing computer analysis are used. Data is parsed to move-level records, capturing clock times, engine evaluations, board state, and move information for every move in every game.

#### 3) Feature Engineering
Move-level features are derived from the raw API data, including time pressure normalisation, eval volatility (rolling standard deviation of engine eval), material complexity scores, castling flags, piece move types, and several interaction terms. A unified eval column combines centipawn and forced-mate evaluations into a single continuous signal.

#### 4) Exploratory Data Analysis
EDA and hypothesis testing are performed on each dataset to examine how error rates vary across time pressure bins, game phase, board complexity, and eval volatility. This informed both feature selection and the decision to model blitz and standard time controls separately.

#### 5) Model Training
Four separate models are trained, one per combination of time control (blitz / standard) and error type (blunder / inaccuracy). Class imbalance to capture the blunder / inaccuracy signal is handled via scale_pos_weight. Models evaluated include XGBoost and CatBoost, with PR AUC as the primary metric given the rarity of errors relative to normal moves.

#### 6) Feature Selection
RFECV is applied using a CatBoost estimator with stratified cross-validation to identify the optimal feature subset for each of the four tasks. Blunder models are compressed to 7–9 features; inaccuracy models retained around 16 features, reflecting the greater difficulty of predicting subtler errors.

#### 7) Hyperparameter Tuning
RandomizedSearchCV with stratified k-fold cross-validation is used to tune each model, optimising for PR AUC.

#### 8) Modular Implementation
The full pipeline is implemented in a modular way: components folder containing data_ingestion.py, data_transformation.py, and model_trainer.py. Pipeline folder containing predict_pipeline.py and train_pipeline.py. Application.py (Flask server and Dash dashboard).

#### 9) Flask + Dash Web Application
The web application allows a user to enter their Lichess username, load their game history, and automatically train personalised models on their data. The dashboard provides two sections of descriptive analytics (error rates by time pressure, game phase, complexity, and volatility) and predictive tools (time trouble threshold analysis, post-game error attribution, and a game risk timeline). The risk timeline accepts any Lichess game URL and plots predicted blunder and inaccuracy risk across every move alongside actual errors.

#### 10) AWS Deployment
The application is deployed to AWS using Elastic Beanstalk and CodePipeline for continuous deployment.

#### Conclusion
This project builds a personalised chess error prediction system, tackling challenges including complex feature engineering and deriving features, capturing the signal of the errors, multiple models across different time controls, and the difficulty of predicting inaccuracies versus blunders. Evaluation prioritises PR AUC over accuracy given the rarity of errors, and percentile-based classification thresholds are used to ensure risk zones are meaningful regardless of the absolute probability scale. The result is a deployable, player-specific tool that connects raw game data to interpretable error risk predictions.