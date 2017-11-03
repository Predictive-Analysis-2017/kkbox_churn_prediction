# kkbox_churn_prediction
* Course project of Predictive Analysis 2017 Fall NYU
* The data set is from Kaggle [WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)
## Attribute Select
#### transaction.csv (last transaction of each member)
- payment_method
- payment_plan_days
- plan_list_price
- is_auto_renew
- is_cancel

#### member.csv
- city
- bd
- register_via

#### user_log.csv
- total_sec

## Feature Generation
select daily logs back to a month from last daily record
#### Partial of each category (avg by month)
- partial_25
- partial_50
- partial_75
- partial_985
- partial_100
- unique

#### Trend of each category (logistic regression)
- trend_25
- trend_50
- trend_75
- trend_985
- trend_100
- trend_unique
- trend_total

#### Continuous subscription interval
- sub_interval
