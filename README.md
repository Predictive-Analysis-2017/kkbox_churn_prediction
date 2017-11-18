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
- partial_unique

#### Trend of each category (logistic regression)
- trend_25
- trend_50
- trend_75
- trend_985
- trend_100
- trend_unique
- trend_total

#### Continuous subscription interval
- sbp_interval

#### Feature from transaction_date and membership_expire_dat
1. Length of current subscription
2. Elapsed time since last renewal
3. Elapsed time since last suspension
4. Average suspension length (in number of days)
5. Average number of days the previous subscriptions are renewed before expiry date
6. The month of contract expiration
