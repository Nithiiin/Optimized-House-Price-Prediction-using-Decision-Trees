# House Price Regression

## Project Overview
This project aims to predict house prices using various features such as lot size, number of rooms, year built, and more. The dataset used is from the [Kaggle House Prices - Advanced Regression Techniques competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Approach
- Initially, we studied the target variable “SalePrice” and noticed that it had a skewed distribution, hence we transformed it into a log scale for normalization purposes.
- For cleaning both numerical and categorical data, we followed the protocol below:
  1. Numerical columns with fewer than 20 NaN values had their NaN values replaced with the median of the column.
  2. Categorical columns with fewer than 20 NaN values had their NaN values replaced with the string “missing”.
  3. Numerical and categorical columns with more than 20 NaN values were dropped.
- We checked the input numerical features for skewness and transformed the features with skewness over 0.50 using a log(1+x) function to reduce their skewness.
- The train dataset was split into a training and validation set using an 80-20 split.
- The final model used for training and testing was the Gradient Boosted Trees model from the TensorFlow Decision Forest (TFDF) library, as it provided the best performance.
- The test data was fed to the trained Gradient Boosted Trees model for prediction.
- Predicted values were stored in a CSV file named “Submission.csv”.

## Implementation Details
### Packages Used
1. `tensorflow_decision_forests`: GradientBoostedTreesModel, pd_dataframe_to_tf_dataset
2. `sklearn.impute`: SimpleImputer
3. `sklearn.ensemble`: RandomForestRegressor
4. `seaborn`: sns heatmap

### Model Architecture and Parameters
- GradientBoostedTreesModel: `metrics="mse"`, `loss='deviance'`, `learning_rate=0.1`

### Pandas to TF DataFrame
- Batch size: 32
- Epochs: 1 (default)

## Experiments
- The correlation matrix of the top 10 important features and the target variable ‘SalePrice’ was visualized using a heatmap.
- Transformation of skewed features with a log(1+x) function significantly reduced the RMS log error from 0.4 to 0.2.
- Attempts to find and delete outliers for the important features did not yield successful results.
- Handling missing values effectively improved model robustness. Features with more than 20 missing values were deleted, while those with fewer missing values were filled with median values.
- The TFDF library's Gradient Boosted Trees model outperformed the RandomForest and Cart models.

## Results
- The evaluation score of the model on the training set was 0.1198.
- The final score on the Kaggle test set was 0.13742.

## Discussion
- Our final model achieved a score of 0.13742 on Kaggle.
- Future work may involve experimenting with an ensemble of different models like Lasso, Gradient Boosted Trees, etc., or creating neural network models like CNNs.
- We also plan to explore better data cleaning and transformation methods to retain valuable information.
- 
