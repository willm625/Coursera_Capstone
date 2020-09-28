# Coursera_Capstone
Project for the Coursera Data Science Capstone course. 

## Introduction
Everyday, a work commuter is susceptible to a car accident, no matter how safe they are. This project will attempt to create a model which can predict the chance of a car collision occuring. Its purpose is provide people driving to work with the safest route possible to their destination. Ultimately, the driver, worker or not, can navigate safer than before.

## Data
Out of the dataset, I will be using attributes such as report number, address type, roadcond, lightcond, speeding, and st_coldesc. The report number would just be for identification. ADDRTYPE tells whether the collision occurred in an alley, block, or intersection. ROADCOND describes the condition of the road. The Speeding column will tell if a car was going over the speed limit or not. The light condition column tells whether it was dark or bright outside, and also if the street light was functioning properly. STCOLDESC tells what the two cars were doing. For each attribute, I could find how likely a collision is for each value. I plan to split the data into train and test, and then find correlations between key features to make a model.

## Methodology
### Explotatory Data Analysis
Data was in a CSV file, so using the read_csv function from the pandas library I converted it into a dataframe. At first, the dataframe had columns with too many NaN values, so for convenience they were deleted. These columns were 'INTKEY', 'EXCEPTRSNCODE', 'EXCEPTRSNDESC', 'INATTENTIONIND', 'PEDROWNOTGRNT', 'SPEEDING'. To the rest of the columns, the NaN values were replaced with the average or most common value in their own column. 

I then took the features which seemed most important to me and made them into a separate dataframe, df2. Df2 consisted of SEVERITYCODE, COLLISIONTYPE, JUNCTIONTYPE, UNDERINFL, WEATHER, ROADCOND, LIGHTCOND, ADDRTYPE, SEVERITYDESC. Many of these columns had values which were not numerical, making machine learning difficult. Thus, I turned each column’s values into indicator columns. First I used pandas’ get_dummies function and passed in the feature. This would give a dataframe which has the values of the certain feature from df2 as columns. The only values the column could have are 1 or 0, indication existence or not. From there I renamed data and it was added back to df2, while the original column was dropped.

Injury Collision had a Pearson coefficient of 0.99 with a p-value of 0.0. This indicates Injury’s involved in collision have a significantly positive correlation with the Severity of an accident.
As you can see in the Box plot, where there are injury collisions the Severity Code is 2.

Property Damage Only Collisions had a Pearson coefficient of -1.0 and a p-value of 0.0. This suggests that Property Damage is significantly negatively correlated with Severity. Thus, the less chance it was just a property damage collision, the higher the severity. As you can see from the Box plot, the severity is lower when the collision only damaged property.

The Block feature, meaning the collision occurred at a block, received a Pearson coefficient of -0.195 and a p-value of 0.0. This indicates a strong negative correlation between the severity of a collision and the chance it occurred at a block. As visualized by the Box plot below, The severity is lower when the collision was at a block, but it had a code of 2 when it was not.

The Intersection feature told if the car accident occured at an intersection or not. The feature  had a Pearson coefficient of 0.199 and a p-value of 0.0. This suggests that Intersection feature is significantly positively correlated with Severity. Thus if a collision occurred at an intersection the higher the severity. As you can see from the Box plot, the severity is higher when the collision was at an intersection.

Parked Car Involved had a Pearson coefficient of -0.305 and a p-value of 0.0. This suggests that it is significantly negatively correlated with Severity. Thus, if a parked car was involved, then the severity of the collision would be low, as seen from the Box plot. The only time a severity of 2 was recorded was when there were no parked cars involved.

The Mid-Block feature had a Pearson coefficient of -0.200 and a p-value of 0.0. This suggests that it is significantly negatively correlated with Severity. Thus, if a collision occurred at a mid-block not at an intersection, then the severity of the collision would be low. The Box plot shows the only time a severity of 2 was recorded was when the collision was not at the feature location.

### Modeling
SEVERITYCODE is a categorical feature, so instead of implementing linear models I looked at classifications types. Specifically, I used two models, one being Logistic Regression and the other Support Vector Machines. Logistic Regression seemed perfect because it estimates values into classes. In this case, I wanted to classify features with their severity code. I used a Support Vector Machine because it also categorizes data. For both models, I set SEVERITYCODE as the y value and the rest of the features as the X values. I used the train-test split function from sklearn with a test size of .2 to divide X and y. 

### Logistic Regression
In order to get good results, I set the logistic regression model with a C value, the value which indicates the inverse of regularization strength, to 0.01. Also, I chose an optimizer of liblinear. After the model predicted its y values, they were compared with the y test values. Using the predict_proba function, which returns estimates for all classes, I was able to see how accurate the model was for each feature. To further check the models accuracy, I used the jaccard similarity score and log loss function. The jaccard similarity score is a number that tells how accurate the model’s predictions were to the test values, on a scale of 0.0 to 1.0. The model received a jaccard index of 1.0. The log loss evaluates the performance of the model where the predicted output is a probability value between 0 and 1. The logistic regression model had a log loss of 0.006, indicating high accuracy.

### Support Vector Machines
The Support Vector Machine used the same x and y values for the logistic regression model. The kernel in a svm shows which function will be used in the model. In this case, I saw rbf, the radial basis function, as the best option. After predicting y values with the x test set, I imported the f1 score function from sklearns. The f1 score is a measure of the model’s accuracy. The score ranges from 0 to 1, 1 indicating strong precision from the model. In this case, our svm model got a f1 score of 1.0.  The jaccard index score of the model was 1.0, meaning the model’s predictions were very close to the actual values.

## Discussion, Results, Recommendations
Some observations I made on this project was that all my features used in the models were indicators. I had to make the features numerical in order to apply any kind of modeling. Perhaps if there were some features with continuous values, I could have used a linear regression model with SEVERITYCODE. However, the results show that logistic regression and support vector machines were the best choice of models, as their accuracy was high.

## Conclusion
In this study, I have analyzed car collisions and their severity in the Seattle Area. Through data analysis, I have proved that the severity of a collision is higher when there are injuries invovled, and when the collision occured at an intersection. On the other hand, property damage only collisions have a low severity. Collisions at blocks, midblocks, and those involving parked cars tend to be less severe as well. I built a logistic regression model and a support vector machine to determine the severity of a collision. These models could be used to predict how likely a collision is to occur depending on location, weather, and the condition of the driver(under the influence). Hopefully, these models can make people drive safer in conditions where severity has proven to be high.


