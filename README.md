# Coursera Capstone: Data Science for Car Collision Analysis
Project for the Coursera Data Science Capstone course. 

## Introduction
Throughout the United States, a significant portion of the population relies on cars for transportation. The objective of this project is to develop a predictive model capable of assessing the likelihood of a car collision. Its primary aim is to offer the safest possible route to a destination. Ultimately, this initiative aims to enhance road safety, benefitting all drivers, whether they are commuting for work or leisure, and ensuring that they can navigate with increased safety compared to current conditions.

## Data
In this analysis, I will focus on several key attributes from the dataset to predict collision likelihood. These attributes include:

1. **Report Number:** This serves as a unique identifier.
2. **Address Type (ADDRTYPE):** Indicates whether the collision occurred in an alley, block, or intersection.
3. **Road Condition (ROADCOND):** Describes the condition of the road.
4. **Speeding:** Indicates whether a vehicle was exceeding the speed limit.
5. **Light Condition:** Describes lighting conditions and the functioning of streetlights.
6. **STCOLDESC:** Provides details about the actions of the vehicles involved.

My approach involves analyzing the collision likelihood for each value within these attributes. I will then split the data into training and test sets and explore correlations between these key features to develop a predictive model.

## Methodology
### Exploratory Data Analysis
I began by loading the dataset, which was stored in a CSV file, into a pandas data frame using the `read_csv` function. Initially, the data frame contained columns with a significant number of NaN (missing) values. To streamline the data and improve its quality, I decided to remove these columns:

- 'INTKEY'
- 'EXCEPTRSNCODE'
- 'EXCEPTRSNDESC'
- 'INATTENTIONIND'
- 'PEDROWNOTGRNT'
- 'SPEEDING'

For the remaining columns, I handled missing values differently. I replaced NaN values with either the average or the most common value found within each respective column, ensuring that the dataset was ready for further analysis.

I selected the most crucial features and created a separate data frame called 'df2,' which included SEVERITYCODE, COLLISIONTYPE, JUNCTIONTYPE, UNDERINFL, WEATHER, ROADCOND, LIGHTCOND, ADDRTYPE, and SEVERITYDESC. Since many of these columns had non-numeric values, it posed a challenge for machine learning. To address this, I transformed each column's values into indicator columns. Initially, I used pandas' 'get_dummies' function for each feature, resulting in a data frame where the values of each feature from 'df2' became columns. These new columns contained only 1s or 0s, representing the presence or absence of a specific value. I then renamed this data and reintegrated it into 'df2,' subsequently dropping the original non-numeric columns.

The Injury Collision had a Pearson coefficient of 0.99 with a p-value of 0.0. This indicates injuries involved in the collision have a significantly positive correlation with the Severity of an accident.
As you can see in the Box plot, where there are injury collisions the Severity Code is 2.

Property Damage Only Collisions had a Pearson coefficient of -1.0 and a p-value of 0.0. This suggests that Property Damage is significantly negatively correlated with Severity. Thus, the less chance it was just a property damage collision, the higher the severity. As you can see from the Box plot, the severity is lower when the collision only damaged property.

The Block feature, meaning the collision occurred at a block, received a Pearson coefficient of -0.195 and a p-value of 0.0. This indicates a strong negative correlation between the severity of a collision and the chance it occurred at a block. As visualized by the Box plot below, The severity is lower when the collision was at a block, but it had a code of 2 when it was not.

The Intersection feature tells if the car accident occurred at an intersection or not. The feature had a Pearson coefficient of 0.199 and a p-value of 0.0. This suggests that the Intersection feature is significantly positively correlated with severity. Thus if a collision occurred at an intersection the higher the severity. As you can see from the Box plot, the severity is higher when the collision is at an intersection.

Parked Car Involved had a Pearson coefficient of -0.305 and a p-value of 0.0. This suggests that it is significantly negatively correlated with Severity. Thus, if a parked car was involved, then the severity of the collision would be low, as seen from the Box plot. The only time a severity of 2 was recorded was when there were no parked cars involved.

The Mid-Block feature had a Pearson coefficient of -0.200 and a p-value of 0.0. This suggests that it is significantly negatively correlated with Severity. Thus, if a collision occurred at a mid-block, not at an intersection, then the severity of the collision would be low. The Box plot shows the only time a severity of 2 was recorded was when the collision was not at the feature location.

### Modeling
SEVERITYCODE is a categorical feature, so instead of implementing linear models, I looked at classification types. Specifically, I used two models, one being Logistic Regression and the other Support Vector Machines. Logistic Regression seemed perfect because it estimates values into classes. In this case, I wanted to classify features with their severity code. I used a Support Vector Machine because it also categorizes data. For both models, I set SEVERITYCODE as the y value and the rest of the features as the X values. I used the train-test split function from sklearn with a test size of .2 to divide X and y. 

### Logistic Regression
To optimize the logistic regression model's performance, I fine-tuned the C value, representing the inverse of regularization strength, to 0.01 and selected the 'liblinear' optimizer. Subsequently, I used the model to predict y values and compared them to the y test values. By employing the 'predict_proba' function, which provides estimates for all classes, I assessed the model's accuracy for each feature. To further evaluate its performance, I utilized the Jaccard similarity score and the log loss function. The Jaccard similarity score quantifies the alignment between the model's predictions and the test values on a scale from 0.0 to 1.0. Impressively, the model achieved a perfect Jaccard index of 1.0. Additionally, the log loss, which assesses the model's performance when predicting probabilities between 0 and 1, was remarkably low at 0.006, affirming its high accuracy.

### Support Vector Machines
The Support Vector Machine used the same x and y values for the logistic regression model. The kernel in an SVM shows which function will be used in the model. In this case, I saw rbf, the radial basis function, as the best option. After predicting y values with the x test set, I imported the f1 score function from sklearns. The f1 score is a measure of the model’s accuracy. The score ranges from 0 to 1, 1 indicating strong precision from the model. In this case, our SVM model got a f1 score of 1.0.  The Jaccard index score of the model was 1.0, meaning the model’s predictions were very close to the actual values.

## Discussion, Results, Recommendations
An important observation during this project was that all the features I used in the models were categorical indicators. To perform any modeling, I needed to convert these features into numerical values. While linear regression could have been an option if there were continuous features, the results indicated that logistic regression and support vector machines were the most suitable models due to their high accuracy.

## Conclusion
In this study, I have analyzed car collisions and their severity in the Seattle Area. Through data analysis, I have proved that the severity of a collision is higher when there are injuries involved, and when the collision occurs at an intersection. On the other hand, property damage-only collisions have a low severity. Collisions at blocks, midblocks, and those involving parked cars tend to be less severe as well. I built a logistic regression model and a support vector machine to determine the severity of a collision. These models could be used to predict how likely a collision is to occur depending on location, weather, and the condition of the driver(under the influence).


