import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv("House Price India.csv")

# print(data.head()) -> first five data 
# print(data.shape) -> get the number of rows and the columns
# print(data.columns) -> get the names of the columns

X = data["living area"]
y = data["Price"]

# print(X.isnull().sum())  -> get the number of missing values in the living area column
# print(y.isnull().sum())  -> get the number of missing values in the Price column

# Feature scaling
mean_X = X.mean()
std_X = X.std()
X_scaled = (X - mean_X) / std_X

cost_history = []
m=X_scaled.size
alpha = 0.01
iterations = 1000
w,b=0,0

for iteration in range(iterations):

    # Prediction
    y_hat = w * X_scaled+ b

    # Cost
    diff = []
    for i in range(m):
        diff.append(y[i] - y_hat[i])

    J = (1 / (2 * m)) * sum(i**2 for i in diff)

    cost_history.append(J)

    # Gradients
    dw,db = 0,0

    for i in range(m):
        dw += -(y[i] - y_hat[i]) * X_scaled[i]
        db += -(y[i] - y_hat[i])

    dw = dw/m
    db = db/m

    # Update parameters
    w = w - alpha * dw
    b = b - alpha * db

    # Print progress
    if iteration % 100 == 0:
        print("Iteration:", iteration, "Cost:", J)

    print("Final w:", w)
    print("Final b:", b)

#Plotting
plt.plot(range(iterations), cost_history)

plt.xlabel("Iterations")
plt.ylabel("Cost J(w,b)")
plt.title("Cost Function vs Iterations")

plt.show()