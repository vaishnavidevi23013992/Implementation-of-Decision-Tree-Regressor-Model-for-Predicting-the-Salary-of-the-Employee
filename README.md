# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VAISHNAVIDEVI V
RegisterNumber: 212223020230
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn import metrics
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![Screenshot 2025-04-12 094112](https://github.com/user-attachments/assets/d56b5aed-0f1b-41c2-b0be-25143d95c9ff)
![Screenshot 2025-04-12 094117](https://github.com/user-attachments/assets/fef634da-2e94-4e5a-bbbd-9080eeb4f9e5)
![Screenshot 2025-04-12 094121](https://github.com/user-attachments/assets/36fb04e9-07d7-4bc3-bc38-48256e9dbdae)
![Screenshot 2025-04-12 094126](https://github.com/user-attachments/assets/29952f2f-6754-495e-898c-f0bdf82e4b67)
![Screenshot 2025-04-12 094130](https://github.com/user-attachments/assets/2810de14-1647-4d26-8892-21477bd76494)
