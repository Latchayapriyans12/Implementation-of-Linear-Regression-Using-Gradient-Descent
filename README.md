# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
#Program to implement the linear regression using gradient descent.
#Developed by: latchaya priyan S
#RegisterNumber:  212224230139


import numpy as  np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  X=np.c_[np.ones(len(X1)),x1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)        
    theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv("/content/50_Startups (1).csv")
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled);

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

*/
```

## Output:
![2ff682ce-9bfb-495c-bc76-7a495d7b8a91-0](https://github.com/user-attachments/assets/49004cd0-3a85-4613-a53f-995c6842d868)
![2ff682ce-9bfb-495c-bc76-7a495d7b8a91-1](https://github.com/user-attachments/assets/8d259fa0-4965-4109-90d1-5fbc440dd64a)
![2ff682ce-9bfb-495c-bc76-7a495d7b8a91-2](https://github.com/user-attachments/assets/299aa047-d567-4484-8715-f15cbdf7a969)
![2ff682ce-9bfb-495c-bc76-7a495d7b8a91-3](https://github.com/user-attachments/assets/5e6ae3d3-7568-4240-a7ae-488cb0e137ac)
![2ff682ce-9bfb-495c-bc76-7a495d7b8a91-4](https://github.com/user-attachments/assets/65a01854-bba4-4b52-bb8c-e19357bc00ab)
![2ff682ce-9bfb-495c-bc76-7a495d7b8a91-5](https://github.com/user-attachments/assets/43779414-2fdc-4edb-bf79-e0a0140d2987)
![2ff682ce-9bfb-495c-bc76-7a495d7b8a91-6](https://github.com/user-attachments/assets/6b3c55cf-978c-4251-b1aa-c5498da2e738)
![2ff682ce-9bfb-495c-bc76-7a495d7b8a91-7](https://github.com/user-attachments/assets/6e1f078f-14f1-4072-957d-92c72435ac18)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
