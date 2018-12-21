# Big-Data-Project
## Predicting the number of Dengue cases

### Project Description
1. The dataset used in this project has data related to two cities: San Juan, Puerto Rico and Iquitos, Peru. Since we hypothesize that the spread of dengue may follow different patterns between the two, we divided the dataset, train seperate models for each city, and then join our predictions.

2. Our target variable, total_cases is a non-negative integer, so we made some numerical predictions. We used standard regression techniques for this type of prediction.

3. Since the mean of the data is much greater than the variance of the data we used negative binomial regression for it.

4. We found correlation between total cases and features and then selected those features which strongly co-relate with our label.

5. We created a feature vector using those features and train our logistic regression model.

6. We used cross validation to estimate our accuracy.

7. We used MAE as evaluation metric.

8. We used normalization to improve accuracy.

9. We used bagging to reduce variance.

10. We followed a simple method to fill the missing values. We used the maximum occuring value of that feature to fill it.

