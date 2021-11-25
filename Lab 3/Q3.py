import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Question 3
df=pd.read_csv("pima-indians-diabetes.csv") #importing csv file as a dataframe
columns = ["pregs","plas","pres","skin","test","BMI","pedi","Age"]
for column in columns: #changing value of outliers with median of data
    iqr=df[column].quantile(0.75)-df[column].quantile(0.25)
    med=df[column].median()          #calculating median and iqr*1.5 for respective column
    for i in range(len(df)):
        #changing outliers(by comparing) value to median of the resp. column
        if df.loc[i,column]<(df[column].quantile(0.25) - 1.5*iqr ) or df.loc[i,column]>(df[column].quantile(0.75)+1.5*iqr):
            df.loc[i,column] = med

# apply Standardization techniques
for column in df.columns:
    df[column] = (df[column] - df[column].mean()) / (df[column].std())

df1 = df[columns]

#Question 3 part (a)
print("Question 3 part (a)")
#Calculating dimension reduced data set(l=2)
pca=PCA(n_components=2)
pca.fit(df1)
x_pca=pca.transform(df1)
x_pca=pd.DataFrame(x_pca)
#converting dataframe to numpy array to calculate Covariance matrix
df1=np.array(df1)
#Returns covariance matrix
Cov_matrix=np.cov(df1,rowvar = False)
#returns eigen value and eigen vectors
eigen_val, eigen_vect = np.linalg.eig(Cov_matrix)
Sort_eigen_val=sorted(eigen_val,reverse=True)
#returns 2 largest eigen value on which data is projected on
print("Eigen values of the given array: \n", Sort_eigen_val[0:2])
#Returns Variance of projected data
print("Variance of the projected data(l=2) :")
print("Col1 =",x_pca.iloc[:,0].std()**2," ","Col2 =",x_pca.iloc[:,1].std()**2)
#ploting scatter plot of reduced dimension less data
plt.scatter(x_pca.iloc[:,0],x_pca.iloc[:,1],marker="o",color="b", alpha=0.33)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Plot of data after dimensionality reduction")
plt.show()

#Question 3 part (b)
print("\nQuestion 3 part (b)")
#Eigen values is calculated in line 42 and sorted in line 43
plt.plot([i for i in range(1,9,1)], Sort_eigen_val)
plt.xlabel("Order of Eigen value")
plt.ylabel("Eigen Value")
plt.title("Plot of Eigen values in descending order")
plt.show()

#Question 3 part (c)
print("\nQuestion 3 part (c)")
Error=[]
N=len(df1)
for i in range(2,9,1):
    pca=PCA(n_components=i)
    pca.fit(df1)
    x_pca = pca.transform(df1)
    X_ori = pca.inverse_transform(x_pca)
    X_error = 0
    for j in range(N):
        sum = 0
        for k in range(8):
            sum += (X_ori[j][k]-df1[j][k])**2
        X_error += sum**0.5
    Error.append(X_error)
    Cov_matrix = np.cov(x_pca, rowvar=False)
    print("For l =", i, "\n", Cov_matrix)

plt.plot([i for i in range(2,9,1)],Error)
plt.xlabel("No. of componenets (l)")
plt.ylabel("Reconstruction Error")
plt.title("Plot of recontruction error vs No. of Components")
plt.show()

#Question 3 part (d)
print("\nQuestion 3 part (d)")
pca=PCA(n_components=8)
pca.fit(df1)
x_pca=pca.transform(df1)
Cov_matrix=np.cov(df1,rowvar = False)
print("\n\n\nCov. Matrix")
print(Cov_matrix)