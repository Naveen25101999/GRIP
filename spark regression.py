# Predict the percentage of an student based on the no. of study hours. 
# This is a simple linear regression task as it involves just 2 variables.

# Objective : predict the percentage criteria based on the number of study hours a student spent daily.
# predict the percentage if a person spent 9.25 hrs/day.

# import required libraries
import pandas as pd # for data manipulation
import numpy as np # numerical python / for numerical calculations
import matplotlib.pyplot as plt # visualization purpose
import seaborn as sns # advanced visualization
from sklearn.model_selection import train_test_split # for data partation
import statsmodels.formula.api as smf # regression module (Simple Linear Regression)
from sklearn import metrics # for accuracy

# load the data
study = pd.read_excel(r"C:\Users\navee\Desktop\intrenshala\spark intrenship\spark data.xlsx")
study

####### E D A ############
# checking for null values
study.isna().sum() # no null values

################################################
#  AutoEda library sweetviz for faster evaluation
import sweetviz as sw
report = sw.analyze(study)
report.show_html()
################################################

# Graphical Representation
study.Hours.plot(kind = 'hist')
study.Scores.plot(kind = 'hist')

# Bivariate analysis Scatter plot
plt.scatter(x = study.Hours, y = study.Scores) # Graph shows linear relationship (+ve correlation)
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

# Box Plot
study.plot(kind = 'box', subplots = True) # No outliers

# splitting data into inputs and output
x = pd.DataFrame(study.Hours) # input variable
y = pd.DataFrame(study.Scores) # Predictor / Output variable

# Data Partation as train and test
# x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)
# basically not required for this SLR but we can use ot for large dataset

############### Simple Linear Regression #######################
# Building Regression model
# model1
model = smf.ols('Scores ~ Hours', data = study).fit()
model.summary() 
# 2_square > 0.8 and 

pred = model.predict(x)
pred

# scatter plot (Actual vs Predicted)
plt.scatter(study.Hours, study.Scores)
plt.plot(study.Hours, pred, 'r')
plt.legend(['Observed data', 'Predicted line'])
plt.show()
# looks like predictions and actual data follows the same pattern

# Error Calculation / Residuals
res = study.Scores - pred
np.mean(res)

res_sqrt = res * res
mse = np.mean(res_sqrt)
rmse = np.mean(mse)
rmse
metrics.mean_absolute_error(study.Scores, pred)

### Making prediction using above model ###
# making predictions for 5 values
pred_data = pd.DataFrame({'Hours': pd.Series([9.25, 9.30, 9.35, 9.40, 10])}) #(we have to predict the scores for a student for 9.25 hrs reading hrs)

pred_d = model.predict(pred_data)
pred_d
   # 91.094854

print(f'Student will get : {pred_d[0]} % if he Study 9.25 hrs/day')










