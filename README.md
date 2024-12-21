# Residency Match and Scramble Prediction

This repository contains a machine learning project aimed at predicting the number of unfilled residency positions in various medical specialties. The project leverages historical data from the National Resident Matching Program (NRMP) and focuses on providing insights for applicants about potential unfilled positions during the SOAP (Supplemental Offer and Acceptance Program) process.

## Overview

The residency match is a critical process for medical school graduates seeking specialization. While many applicants are matched to residency programs, some remain unmatched and participate in the SOAP process to secure unfilled positions. This project uses supervised machine learning to predict the number of unfilled positions in each specialty before Match Day, helping applicants better prepare for the scramble process.

### Key Goals

- Predict the number of unfilled residency positions per specialty.
- Provide insights to applicants about potential opportunities in the SOAP process.
- Reduce uncertainty for unmatched applicants by offering predictions based on historical trends.

---

## Data Overview

### Source
The data was obtained from NRMP's yearly reports on the Main Residency Match Data, spanning the years 2015–2024. Significant preprocessing and feature engineering were performed to create a comprehensive dataset suitable for machine learning.

### Original Features
The raw data included attributes such as:

- Specialty
- No Programs Positions Offered
- Unfilled Programs
- US Senior Applicants
- Total Applicants
- No of US Senior Matches
- No of Total Matches
- % Filled by US Seniors
- % Filled Total
- Ranked Positions by US Seniors
- Total Ranked Positions


### Engineered Features
To enhance predictive power, the following features were derived:

| Feature                              | Calculation                                                                                   | Significance                                                                                     |
|--------------------------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Applicants per Position Ratio        | Total Applicants to Specialty / Offered Positions for Specialty                              | Measures competition level in the specialty.                                                   |
| Programs to All Programs Ratio       | Number of Programs for Specialty / Total PGY-1 Programs                                       | Indicates the specialty's commonness.                                                          |
| US MD Senior to Total Applicants Ratio | US MD Senior Applicants to Specialty / Total Applicants to Specialty                          | Shows the specialty's desirability among preferred applicants.                                  |
| Average Program Size                 | Offered Positions for Specialty / Number of Programs for Specialty                           | Reflects the average size of programs within the specialty.                                     |
| Applications to All Applications Ratio | Total Applicants to Specialty / Total PGY-1 Applicants                                       | Gauges the popularity of the specialty.                                                        |
| Year                                 | Year of the data                                                                             | Captures temporal trends.                                                                       |

---

## Machine Learning Approach

### Libraries Used
- **Data Handling and Visualization**: `pandas`, `matplotlib`
- **Preprocessing and Modeling**: `scikit-learn`
- **Metrics**: `mean_squared_error`, `r2_score`

### Pipeline
1. **Data Importing and Cleaning**:
   - Combined datasets from 2015–2024.
   - Removed irrelevant or redundant features and entries (e.g., positions with `0` offered).
2. **Feature Engineering**:
   - Created meaningful features from raw data to improve predictive capability.
3. **Modeling**:
   Five models were trained and evaluated for predicting unfilled positions:

   - **Decision Tree Regressor**:
     A grid search was used to tune hyperparameters such as `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`. The model was trained and validated using 5-fold cross-validation, with the best estimator evaluated on the test data.

   - **Random Forest Regressor**:
     This ensemble method involved tuning `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`. Like the Decision Tree model, the Random Forest Regressor used grid search with 5-fold cross-validation to identify the best parameters.

   - **K-Nearest Neighbors (KNN)**:
     A pipeline was built with data scaling (`StandardScaler`) and the `KNeighborsRegressor`. The hyperparameter tuning process focused on `n_neighbors`, `weights`, and `algorithm`. Grid search with cross-validation ensured the optimal combination of parameters.

   - **Neural Network (MLP Regressor)**:
     A pipeline including `StandardScaler` and `MLPRegressor` was created. Hyperparameters like `hidden_layer_sizes` and `activation` functions were optimized through grid search. The model was trained for a maximum of 1000 iterations.

   - **Linear Regression**:
     As a baseline model, Linear Regression was implemented and evaluated alongside the other methods.

   Each model's predictions were evaluated using:
   - **Mean Squared Error (MSE)**
   - **R² Score**

4. **Model Evaluation and Comparison**:
   - A comparison was made across the models using both MSE and R² scores.
   - Visualizations were created to compare actual versus predicted values for each model, highlighting performance differences.

5. **Cross-Validation Score Comparison**:
   Cross-validation scores for each model were computed and normalized by the number of data points. These scores were visualized as a bar chart to compare model performance during cross-validation.

---

## Usage

### Requirements
- Python 3.8+
- `pandas`, `matplotlib`, `scikit-learn`

### Steps to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/residency-match-prediction.git
