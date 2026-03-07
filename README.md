# Big-Data-A2

Big-Data-A2 is a data analysis and visualization project for a Big Data assignment. The project uses Python and Jupyter Notebook to explore, analyze, and visualize datasets relevant to the assignment objectives.

## Project Overview

- **Purpose:**  
  Predict NYC taxi trip tips and passenger tipping behavior using machine learning models trained on real taxi trip data (January 2024).
- **Features:**  
  - Automated data download and preprocessing from NYC Taxi & Limousine Commission
  - Feature engineering (temporal, trip, fare, and zone features)
  - Multiple machine learning models: Linear Regression, Random Forest (both regression and classification), and Neural Networks
  - Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
  - Interactive Streamlit dashboard for data exploration and visualization
  - Model evaluation on test set with comprehensive metrics

## Requirements

- Python 3.8 or higher
- Jupyter Notebook
- Required libraries listed in `requirements.txt`

## Setup Instructions

1. Clone the repository and navigate to the project directory:
    ```bash
    cd Big-Data-A1
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. **Option A: Run the full analysis (recommended for first-time use)**
   - Launch Jupyter Notebook and open `assignment1.ipynb`:
     ```bash
     jupyter notebook assignment1.ipynb
     ```
   - Run all cells to download data, preprocess, perform feature engineering, and train models
   - This will generate processed data in `data/processed/cleaned_trips.parquet`

4. **Option B: Run the interactive dashboard**
   - Once data preprocessing is complete, launch the Streamlit app:
     ```bash
     streamlit run app.py
     ```
   - Explore interactive visualizations and apply filters to the NYC taxi data

## Files

- `assignment1.ipynb`: Main Jupyter Notebook containing:
  - Data download and validation
  - Data cleaning and preprocessing
  - Feature engineering (temporal, trip, fare, and zone features)
  - Target variable creation (tip_amount and high_tip classification)
  - Train/validation/test split
  - Feature scaling with StandardScaler
  - Multiple ML models: Linear Regression, Random Forest Regressor/Classifier, and Neural Network
  - Hyperparameter tuning with RandomizedSearchCV
  - Model evaluation and visualization

- `requirements.txt`: List of Python dependencies (23 packages)

- `data/`: Directory containing:
  - `raw/`: Downloaded raw data files (yellow_tripdata_2024-01.parquet, taxi_zone_lookup.csv)
  - `processed/`: Cleaned and engineered data (cleaned_trips.parquet)

## Usage

### Full Workflow
1. **Run the Jupyter Notebook** (`assignment1.ipynb`):
   - Downloads NYC taxi trip data (January 2024) and taxi zone lookup data automatically
   - Cleans and validates the data
   - Engineers features for machine learning
   - Trains and evaluates multiple models (Linear Regression, Random Forest, Neural Network)
   - Performs hyperparameter tuning on classification models
   - Outputs evaluation metrics and visualizations

2. **Launch the Streamlit Dashboard** (`app.py`):
   ```bash
   streamlit run app.py
   ```
   - Provides interactive data exploration
   - Filter by hour range, day of week, and payment type
   - View 6 different visualizations of the NYC taxi data
   - 
## Machine Learning Models
### Target Variables
- **Regression Task**: Predict `tip_amount` (continuous value)
- **Classification Task**: Predict `high_tip` (binary: 1 if tip > 20% of fare, else 0)

### Models Trained
1. **Linear Regression** - Baseline regression model
2. **Random Forest Regressor** - Ensemble regression model with hyperparameter tuning
3. **Random Forest Classifier** - Ensemble classification model with hyperparameter tuning
4. **Logistic Regression** - Linear classification model
5. **Neural Network** - Feedforward neural network with 2 hidden layers (PyTorch)

### Model Evaluation
Models are evaluated on test set (25% of data) using:
- **Regression Metrics**: MAE, RMSE, R² Score
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Key Features Used
- **Temporal**: pickup_hour, pickup_day_of_week, is_weekend
- **Trip Characteristics**: trip_duration_minutes, trip_speed_mph, trip_distance
- **Fare Analysis**: fare_per_mile, fare_per_minute, log_trip_distance
- **Location**: PULocationID, DOLocationID, pickup/dropoff zones

This project is for educational purposes as part of a Big Data assignment.