# kkbox_churn_prediction
* Course project of Predictive Analysis 2017 Fall NYU
* The data set is from Kaggle [WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)

In this project, we analyzed the the data from KKbox and predicted whether an user will churn after a short period (i.e. 30 days) from the expiration of the current service subscription. We pre-processed the data and uncovered predictive features from it. Then we applied some of the classification algorithms to train the models, evaluated the performance and analyzed their discrepancies. By tuning these models, we made best prediction by Decision Tree model and got high prediction accuracy at 95.2%. More detail please refer to [report](https://github.com/Predictive-Analysis-2017/kkbox_churn_prediction/blob/master/Churn%20prediction%20for%20KKBOX%20music%20streaming%20service.pdf)

## Data Understanding
The dataset contains about 5 million members data, 21 million transaction data and more than 0.4 billion user log from February 2017 to April 2017. Total size of data is 32.7 GB

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

## Accuracy Result
The first time we just run the features from member profile and transactions, the modeling by decision tree is 90.2%, which is already very good. The reason is that the feature _is_cancal_ and _is_auto_renew_ are highly correlated to predicting result.
We tried to improve it further by adding new generated temporal features from subscriptions.csv.(Referenced by [paper](https://github.com/Predictive-Analysis-2017/kkbox_churn_prediction/blob/master/paper/Churn_prediction_in_subscription_service.pdf)) The accuracy increased to 95.16%.

| Decision Tree 	| Random Forest 	| SVM    	|
|---------------	|---------------	|--------	|
| 95.16%        	| 90.59%        	| 92.83% 	|

## To do
#### Add new features
#### Trend of music category (logistic regression)
- trend_25
- trend_50
- trend_75
- trend_985
- trend_100
- trend_unique
- trend_total
