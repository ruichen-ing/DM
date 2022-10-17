import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# X: design matrix
def DesignMatrix(X):
    lst = []
    for i in range(X.shape[0]):
        row = [1]
        for j in range(X.shape[1]):
            row.append(X.iloc[i][j])
        lst.append(row)
    dm = np.mat(lst)
    return dm

def regression(X, y):
    X_reverse = np.linalg.inv(X.T@X)
    β = X_reverse@X.T@y
    return β

# I: data set
def MSE(β, dm, y):
    sum = .0
    for i in range(dm.shape[0]):
        sum = sum + (y[i] - dm[i]@β)**2
    return sum / dm.shape[0]



if __name__ == '__main__':
    data = pd.read_csv('./data/no2_dataset.csv')

    # print(data)

    plt.figure(3, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(data['cars_per_hour'], data['no2_concentration'], color="red")
    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[0])

    plt.subplot(1, 3, 2)
    plt.scatter(data['wind_speed'], data['no2_concentration'], color="blue")
    plt.xlabel(data.columns[2])
    plt.ylabel(data.columns[0])

    plt.subplot(1, 3, 3)
    plt.scatter(data['wind_direction'], data['no2_concentration'], color="green")
    plt.xlabel(data.columns[3])
    plt.ylabel(data.columns[0])

    X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:, 1:4], data['no2_concentration'], random_state=10,
                                                        test_size=0.2)
    dm = DesignMatrix(X_train)

    dm_multi = dm.T @ dm
    # series to array
    y_arr = np.stack(Y_train.values)
    # transpost row array to column array(equal to vector), -1: as many rows as needed
    y = y_arr.reshape(-1, 1)
    β = regression(dm, y)
    # print(β)
    # print(β.shape)

    print(MSE(β, dm, y))