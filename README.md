# Task 1: Data Cleaning & Preprocessing

This repository contains the solution for Task 1 of the Elevate Labs AI & ML Internship. The script `data_preprocessing.py` performs a full data cleaning and preprocessing pipeline on the Titanic dataset.

## Steps Performed:

1.  **Load Data**: The Titanic dataset was loaded into a pandas DataFrame.

2.  **Handle Missing Values**:
    * Missing `Age` values were filled with the median age.
    * Missing `Embarked` values were filled with the mode.
    * The `Cabin` column was dropped due to a high number of missing values.

3.  **Encode Categorical Features**:
    * The `Sex` column was converted to numerical values (0 for male, 1 for female).
    * The `Embarked` column was one-hot encoded to create separate binary columns.

4.  **Feature Scaling**:
    * Numerical features like `Age` and `Fare` were standardized using `StandardScaler` to bring them to a similar scale.

5.  **Outlier Removal**:
    * Outliers in the `Age` and `Fare` columns were detected using the IQR method and removed from the dataset. Boxplots were generated to visualize the data before and after this process.

The final, cleaned dataset is saved as `cleaned_titanic.csv`.
