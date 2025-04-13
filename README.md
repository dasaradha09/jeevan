# **Student's Dropout Prediction using Supervised Machine Learning Classifiers**
A Data science project on Predicting Students' dropout using Machine Learning classification models

# **Introduction**
The goal of this project is to develop a predictive model using machine learning classification algorithms to identify students who are likely to drop out. By leveraging data on student demographics, academic performance, socio-economic factors, and other relevant variables, the aim is to build a robust predictive model that can effectively forecast the likelihood of students dropping out.

The data gathered are from both internal and external sources, pulling information from different institutions. This data was drawn from various databases within universities and colleges. It covers student records spanning from the academic year 2008/2009 to 2018/2019.

This model can then be used by educational institutions to allocate resources, implement early intervention strategies, and support at-risk students.

# **Problem**
In today's educational landscape, student retention and success are of utmost importance for educational institutions. Identifying students who are at risk of dropping out and implementing timely interventions can greatly contribute to improving graduation rates and ensuring academic success. 

# **Methodology Approach**
Data collected through Kaggle datasets.
Data processing and descriptive analysis
Exploratory Data Analysis using Python visualization tools to gain insights into the data and identify any patterns or trends.
Predictive analysis using Machine Learning Classification algorithms.

# **Stack used**
**Libraries**
**Data Wrangling and processing Libraries**
Pandas
Numpy
**Visualization Libraries**
Matplotlib
Plotly express
Plotly Graph Objects
**Machine Learning Libraries**
Scikit Learn


# **Cleaning the Data**
**Handle Missing Values**

Fill with mean/median/mode or remove rows.

**Remove Duplicates**

Drop repeated records.

**Fix Inconsistencies**

Standardize formats (e.g., "Male", "male" → "Male").

**Remove Outliers**

Use IQR or z-score to filter unusual data points.

**Encode Categorical Data**

Convert text to numbers (Label or One-Hot Encoding).

**Feature Scaling**

Normalize or standardize numerical features.

**Clean Text (if needed)**

Lowercase, remove stopwords, and apply lemmatization.
![image](https://github.com/user-attachments/assets/8c8c6e1d-8b7a-44ea-ab58-cd177baae5f9)

# **EXPLORING THE DATA**

The majority of categorical variables in the downloaded dataset have already been converted to numerical format. However, for the purpose of exploratory data analysis (EDA), we will revert certain columns to their original categorical form.
**Target feature**
From the target column we can infer the following:

Dropout: The student dropped out
Graduate: The student graduated
Enrolled: The student is currently enrolled

**Student Status Distribution (Pie Chart Explanation)**
The pie chart shows the proportion of students based on their current academic status:

Dropout – Represents the students who discontinued their studies.

Graduate – Indicates students who successfully completed their program.

Enrolled – Refers to students who are still studying.

![image](https://github.com/user-attachments/assets/01a34e25-6eea-443a-afbe-f0525f779cb7)

# **Gender distribution of students**

![image](https://github.com/user-attachments/assets/7f182fed-10e0-435d-a2c3-a2a7343147a4)

**Observation**
Majority of the students are enrolled in Nursing and it also had the lowest droput rate of about 15.4%.

The course that had the highest dropout rate was Biofuel Production Technologies (66.7%) which is also had the least number of enrolled students followed by Equiniculture (55.3%).
**Gender Percentage by count of students**

![image](https://github.com/user-attachments/assets/d7b7c7bd-4db7-4aeb-9258-067301a74b6e)


**Observation**

There was a significant number of female students (64.8%) compared to the males (35.2%).
Also it is observed that there was a higher rate of dropout students that were male (45.1%), compared to the females (25.1%).

**Students Enrolled courses**
![image](https://github.com/user-attachments/assets/1644f543-fa6c-456e-8403-0e6030765779)


**Student's Financial Status**
Next, we will analyze the financial status of students to understand how scholarship status, debt status, and tuition payment status correlate with dropout rates.
![image](https://github.com/user-attachments/assets/6f4e1645-ee64-4486-bc58-20bd109f7232)*

**Observation**
Unsurprisingly, students who were in debt and had not completed payment for tuition had a higher dropout rate of 62% and 86.6% respectively.

Similarly, students who were granted scholarships had a low dropout rate of 12.2% compared to those who were not given (38.7%).

Based on our EDA, we have a good understanding of how the data is distributed by gender and age, as well as how certain features like courses enrolled and financial status correlate with the dropout rate. In the next phase, we will build and train a classification model to see if our findings are consistent with the model's results.
# **Data Processing**
Before standardizing the data we need transform the target feature from categorical to numerical data.
![image](https://github.com/user-attachments/assets/414af3ad-e4af-4f5c-bc63-ab29315ba7d8)
![image](https://github.com/user-attachments/assets/878b89fc-7c7d-4c84-a31b-a6421ba88ce0)
After normalizing the data
![image](https://github.com/user-attachments/assets/93930de8-fd77-45d2-9f4e-e1ae57103c4e)

# **Model Building**
Because the target variable is categorical (either Dropout or not), this is a classification problem. We will train five supervised machine learning classification models:

Logistic regression
Decision trees
Support vector machines
Random forest
K-nearest neighbors
First, we will split our data into training, validation, and test sets. We will set the random state to 42 to ensure reproducibility.

**Logistic regression**
Accuracy of logistic regression model on the training set is 83.2%
Accuracy of logistic regression model on the validation set is 84.3%
**Decision trees**
From the training curve we can estimate that the best depth value that yielded the highest accuracy score was between 2 and 5. However, a better way to pick out the depth value would be to get the index value that corresponds to the highest accuracy score in the validation_acc list.

**Support vector machines**

Accuracy of Support vector classifier model is 80.9%
To train both Random forest and K-nearest neighbours, we will make use of Grid Search to determine the best parameters. We will also stick to the default value for cross validation which is 5 folds

**Random forest**
With default parameters: 
Accuracy of Random forest model on training data is 99.9%
Accuracy of Random forest model on validation data is 85.5%
We can perform a grid search to generate best parameters to train a random forest model.
Accuracy of Random forest model on test data is 82.8%


**K- Nearest Neighbours**
With Grid search best estimator parameters: 
Accuracy of KNN model on training data is 83.89999999999999%
Accuracy of KNN model on validation data is 82.3%


**Accuracy for each classification model**


![image](https://github.com/user-attachments/assets/cf6f2b45-f6a5-46e6-9a2e-ed764c3793e2)

# **RESULTS**
**Feature Importances**
An AUC score of 0.78 means that the model is able to correctly classify 78% of the positive cases and 22% of the negative cases. This is considered to be good performance, but not excellent. There is still room for improvement

Finally i trained the machine learning by approaching the some regression algorithms and i experienced different type of results and finally random forest and decisioon tree alogorithm has evaluated the high accuracy to predict the inputs by training the model.

![{E2A63DBC-27FB-4E26-B5EA-91F78FBC34DD}](https://github.com/user-attachments/assets/140087d4-4979-4f89-9864-ad5d1c398916)

According to the result if the student is dropped from the school the alert message is sent to the parents mail or phone number .








