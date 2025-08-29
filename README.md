# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the dataset (e.g., student_scores.csv) using a library like pandas and extract the independent variable (e.g., Hours) and dependent variable (e.g., Scores) into arrays for processing.

2.Train-Test Split: Divide the dataset into training and testing sets using a function like train_test_split from scikit-learn, typically with a specified test size (e.g., 1/3) and random state for reproducibility.

3.Model Training: Initialize a linear regression model using LinearRegression from scikit-learn and train it on the training data (x_train, y_train) to learn the relationship between the variables.

4.Prediction: Use the trained model to predict the dependent variable (y_pred) for the test set (x_test) and evaluate the model's performance using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

5.Visualization and Evaluation: Plot the training and test data points along with the regression line using a plotting library like matplotlib to visualize the model's fit, and compute performance metrics (MSE, MAE, RMSE) to assess accuracy.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: FRANKLIN.F
RegisterNumber:212224240041
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
## HEAD

<img width="340" height="267" alt="image" src="https://github.com/user-attachments/assets/4e8c0649-2309-4043-96c2-109ff06ebe19" />


## TAIL

<img width="313" height="256" alt="image" src="https://github.com/user-attachments/assets/81cbbeca-0fea-4f23-a9f8-2ce8b3bb209c" />

## X VALUE

<img width="625" height="560" alt="image" src="https://github.com/user-attachments/assets/8e1df5bb-dacc-4def-b88c-1d190cff89ed" />

## Y VALUE

<img width="870" height="65" alt="image" src="https://github.com/user-attachments/assets/cf71debb-3448-48b4-8061-266f06862033" />

## PRED

<img width="734" height="86" alt="image" src="https://github.com/user-attachments/assets/b589a8cb-09d3-4229-9547-14db17a87bae" />


## TEST

<img width="639" height="47" alt="image" src="https://github.com/user-attachments/assets/56e8f11d-8d8c-462c-a955-285e61dbb171" />


## TRAINING

<img width="1233" height="584" alt="image" src="https://github.com/user-attachments/assets/f7c80233-c273-4626-bb5e-a1dab74474bf" />


## TESTING

<img width="818" height="566" alt="image" src="https://github.com/user-attachments/assets/f30844e0-226e-48b9-a8b0-136328fa568e" />

## MSE MAE RMSE

<img width="351" height="95" alt="image" src="https://github.com/user-attachments/assets/9a892b89-89bf-460b-b846-dd64098cc5d5" />





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
