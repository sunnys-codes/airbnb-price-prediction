# titanic-machine-learning
This repository contains my analysis and machine learning model to predict Titanic survivors using the famous **Titanic Dataset** from [Kaggle](https://www.kaggle.com/c/titanic).

## Dataset Overview

The Titanic dataset consists of two CSV files:
- **train.csv**: Contains passenger information, including whether they survived or not.
- **test.csv**: Contains similar passenger information, but without the survival column (which we will predict).

## Steps Performed

1. **Loaded the data** using `pandas`.
2. **Checked for missing values** and analyzed the shape and structure of the data.
3. **Filled missing values**:
   - `Age`: Filled with the **mean**.
   - `Embarked`: Filled with the **most frequent value** (train data only).
   - `Fare`: Filled with the **median** (test data only).
4. **One-hot encoded** the `Sex` column to convert it into a numerical feature.
5. **Visualized the Age distribution** using a histogram plot.
6. Prepared the data for **machine learning** (in future steps).
