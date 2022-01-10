#Multiple Linear Regression Program
#Sharandeep Singh

from contextlib import suppress
from os import error
import pandas as pd
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
df = pd.read_csv("heart.data.csv")
data = pd.DataFrame(df)
n=498

# INDEPENDENT VARIABLES
# Here, number of Independant variables = 2
q=2
x1 = data.iloc[:,1].values
x2 = data.iloc[:,2].values  
# DEPENDENT VARIABLES
y  = data.iloc[:,3].values

# independent variables matrix
rows,cols = (n,q+1)
x = [[0]*cols]*rows
x_1 = np.array(x)
x_arr = x_1.astype(float)
for i in range(n):
    x_arr[i][0] = 1

for i in range(0,n):
    for j in range(1,q+1):
        x_arr[i][j] = data.iat[i,j]
x_trans = np.transpose(x_arr)
print("\n\t\tX Matrix\n\n",x_arr, "\n")
print("  Dimensions: ",x_arr.shape,"\n\n")
print("\n\t\tX Transpose\n\n", x_trans,"\n")
print("  Dimensions: ",x_trans.shape,"\n\n")

plt.subplot(1,2,1)
plt.scatter(x1,y, color='g')
plt.title("Plot of Heart Disease vs Biking")
plt.ylabel("Heart Disease")
plt.xlabel("Biking")
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(x2,y, color='r')
plt.title("Plot of Heart Disease vs Smoking")
plt.ylabel("Heart Disease")
plt.xlabel("Smoking")
plt.grid(True)
plt.show()


def calculating_R(x1, x2):
    productxy = 0
    sumx1 = 0
    sumx2 = 0
    sumsquaredx1 = 0
    sumsquaredx2 = 0

    for i in range(n):
        sumx1 += x1[i]
        sumx2 += x2[i]
        sumsquaredx1 += x1[i]*x1[i]
        sumsquaredx2 += x2[i]*x2[i]
        productxy += x1[i]*x2[i]


    r = ((n*productxy) - (sumx1*sumx2))/(math.sqrt((n*sumsquaredx1) - (sumx1*sumx1))*(math.sqrt((n*sumsquaredx2) - (sumx2*sumx2))))
    return r

print("  R-Value:   ",calculating_R(x1,x2),"\n")

def calculating_vif(r):
    vif = 1/(1-(r*r))
    return vif

print("  VIF-Value: ",calculating_vif(calculating_R(x1,x2)),"\n")




def b_matrix(x_matrix,x_matrix_transpose,y_matrix,no_predictors):
    print("\n\n  B MATRIX \n")

    # Now, we know B-MATRIX = (X(t)*X)^-1 * (X(t)*Y)
    # Let, B-Matrix = A * B
    # Where A = (X(t)*X)^-1 and B = (X(t)*Y)
    np.set_printoptions(suppress=True)
    x_trans_x_arr = np.matmul(x_matrix_transpose,x_matrix)
    A = np.linalg.inv(x_trans_x_arr)
    B = np.matmul(x_matrix_transpose,y_matrix)

    b_arr = np.matmul(A,B)
    print(b_arr[:,None], "\n")
    print("Dimensions: ",b_arr.shape,"\n\n")
    print("Here, B0 = ", b_arr[0].round(4))
    for i in range(1,q+1):
        print(f"Value of B{i}: ",b_arr[i].round(4))
    print("\n\n")
    return b_arr

b_mat_1 = b_matrix(x_arr,x_trans,y,q)
b_mat = np.transpose(b_mat_1)



# Error Matrix
np.set_printoptions(threshold=False,suppress=True)
first_term = np.matmul(x_arr,b_mat.transpose())
error_mat = []
for i in range(n):
    error_mat.append(y[i] - first_term[i])
error_matrix = np.array(error_mat)
print("\n  Error Matrix\n\n",error_matrix[:,None],"\n")
print("Dimensions: ",error_matrix.shape,"\n")


# Predicted Y
predicted_y = np.matmul(x_arr,b_mat)
print("\n  Y-Matrix\n\n",predicted_y[:,None],"\n")
print("Dimensions: ", predicted_y.shape,"\n")


def predictor():
    print("\n\tPredictor Model\n")
    a = float(input("Enter Value of Biking : "))
    b = float(input("Enter Value of Smoking : "))
    predicted_y_y = b_mat[0] + b_mat[1]*a + b_mat[2]*b     
    print(f"\nThe Value of Heart Disease for given values of Biking and Smoking is : {predicted_y_y}\n")
predictor() 

# plt.subplot(1,2,1)
# plt.scatter(x1,y, color='r', marker='o',label='Plot of Heart Disease vs Biking')
# plt.plot(x1,predicted_y, color='b',label='Regression Line 1')
# plt.legend()
# plt.grid(True)

# # plt.subplot(1,2,2)
# # plt.scatter(x2,y, color='g', marker='o',label='Plot of Heart Disease vs Smoking')
# # plt.plot(x2,predicted_y,color='black', label='Regression Line 2')
# # plt.legend()
# # plt.grid(True)
# plt.show()




