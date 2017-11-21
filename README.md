# kkbox_churn_prediction
* Course project of Predictive Analysis 2017 Fall NYU
* The data set is from Kaggle [WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)
## Attribute Selection
#### transaction.csv (last transaction of each member)
- payment_method
- payment_plan_days
- plan_list_price
- is_auto_renew
- is_cancel
- month of subscription

#### member.csv
- city
- bd
- register_via

## Feature Generation
User listening behavior features, trace daily logs back to 60 days from last daily record from user_log.csv
#### Partial of each category (avg)
- partial_25
- partial_50
- partial_75
- partial_985
- partial_100
- partial_unique
- total_sec
- partial_25_standard deviation
- partial_50_standard deviation
- partial_75_standard deviation
- partial_985_standard deviation
- partial_100_standard deviation
- partial_unique_standard deviation
- total_sec_standard deviation

#### Trend of each category (logistic regression)
- trend_25
- trend_50
- trend_75
- trend_985
- trend_100
- trend_unique
- trend_total

User subscription behavior, from transactions.csv
#### Continuous subscription interval
- elapsed days since last renewal
- elapsed days since last suspension
- days of renewal before expiration
- elapsed days since last renewal_mean
- elapsed days since last suspension_mean
- days of renewal before expiration_mean
- elapsed days since last renewal_standard deviation
- elapsed days since last suspension_standard deviation
- days of renewal before expiration_standard deviation
