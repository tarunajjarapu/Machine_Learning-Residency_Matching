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
- **Metrics**: `r2_score`

### Pipeline
1. **Data Importing and Cleaning**:
   - Combined datasets from 2015–2024.
   - Removed irrelevant or redundant features and entries (e.g., positions with `0` offered).
2. **Feature Engineering**:
   - Created meaningful features from raw data to improve predictive capability.
3. **Modeling**:
   - Used `MLPRegressor` (Multi-layer Perceptron) for regression tasks.
   - Evaluated using `r2_score` to measure model performance.

---

## Usage

### Requirements
- Python 3.8+
- `pandas`, `matplotlib`, `scikit-learn`

### Steps to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/residency-match-prediction.git
