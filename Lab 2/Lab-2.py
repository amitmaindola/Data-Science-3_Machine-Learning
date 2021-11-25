# Amit Maindola (B20079)
# Lab Assignment 2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_file1 = pd.read_csv("landslide_data3_miss.csv")
df1 = df_file1.copy()
df1_shape1 = df1.shape


# Question 1

columns = ["dates","stationid","temperature","humidity","pressure","rain","lightavgw/o0","lightmax","moisture"]
plt.figure(figsize=(6,4))
plt.bar(columns,df1.isnull().sum())
plt.xticks(rotation=25)
plt.show()


# Question 2 (Part a)

df1.dropna(subset=["stationid"], inplace=True)
df1_shape2=df1.shape
print("No. of tuples deleted (in part a) : ", df1_shape1[0] - df1_shape2[0])


# Question 2 (Part b)

thresh = (df1_shape1[1]*2/3)+1
df1.dropna(thresh = thresh, inplace=True)
df1_shape3 = df1.shape
print("No. of tuples deleted (in part b) : ", df1_shape2[0] - df1_shape3[0])


# Question 3

print("\nNo. of missing values in each attribute : ")
print(df1.isnull().sum())
print("No. of missing values in the file : ",df1.isnull().sum().sum())


#Question 4 (Part a)
# Part i
print("\nAfter filling missing values with mean : \n")
df1_fill_mean = df_file1.fillna(df_file1.mean(numeric_only=True)) # filling missing values with mean
df_file2 = pd.read_csv("landslide_data3_original.csv") #original file
#mean comparision
print("\nMean comparision")
print(df1_fill_mean.mean(numeric_only=True))
print(df_file2.mean(numeric_only=True))
#median comparision
print("\nMedian comparision")
print(df1_fill_mean.median(numeric_only=True))
print(df_file2.median(numeric_only=True))
#mode comparision
print("\nMode comparision")
print(df1_fill_mean.mode(numeric_only=True))
print(df_file2.mode(numeric_only=True))
#standard deviation comparision
print("\nStandard deviation comparision")
print(df1_fill_mean.std(numeric_only=True))
print(df_file2.std(numeric_only=True))


#Question 4 (Part a)
# Part ii
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).sum()
print("\n\nRMSE : \n")
rmsTemp=(rmse(np.array(df1_fill_mean["temperature"]), np.array(df_file2["temperature"]))/df_file1["temperature"].isnull().sum())**0.5
rmsHum=(rmse(np.array(df1_fill_mean["humidity"]), np.array(df_file2["humidity"]))/df_file1["humidity"].isnull().sum())**0.5
rmsPres=(rmse(np.array(df1_fill_mean["pressure"]), np.array(df_file2["pressure"]))/df_file1["pressure"].isnull().sum())**0.5
rmsRain=(rmse(np.array(df1_fill_mean["rain"]), np.array(df_file2["rain"]))/df_file1["rain"].isnull().sum())**0.5
rmsLavg=(rmse(np.array(df1_fill_mean["lightavgw/o0"]), np.array(df_file2["lightavgw/o0"]))/df_file1["lightavgw/o0"].isnull().sum())**0.5
rmsLmax=(rmse(np.array(df1_fill_mean["lightmax"]), np.array(df_file2["lightmax"]))/df_file1["lightmax"].isnull().sum())**0.5
rmsMois=(rmse(np.array(df1_fill_mean["moisture"]), np.array(df_file2["moisture"]))/df_file1["moisture"].isnull().sum())**0.5

column1 = ["temperature","humidity","pressure","rain","lightavgw/o0","lightmax","moisture"]
plt.figure(figsize=(6,6))
plt.bar(column1,[rmsTemp,rmsHum,rmsPres,rmsRain,rmsLavg,rmsLmax,rmsMois]);
plt.xticks(rotation=25)
plt.show()


#Question 4 (Part b)
# Part i
df1_interpolate = df_file1.interpolate() # filling missing values using interpolate
#mean comparision
print("\nAfter Interpolation : \n")
print("\nMean comparision")
print(df1_interpolate.mean(numeric_only=True))
print(df_file2.mean(numeric_only=True))
#median comparision
print("\nMedian comparision")
print(df1_interpolate.median(numeric_only=True))
print(df_file2.median(numeric_only=True))
#mode comparision
print("\nMode comparision")
print(df1_interpolate.mode(numeric_only=True))
print(df_file2.mode(numeric_only=True))
#standard deviation comparision
print("\nStandard deviation comparision")
print(df1_interpolate.std(numeric_only=True))
print(df_file2.std(numeric_only=True))


#Question 4 (Part b)
# part ii
def rmse(predictions, targets):
    return ((predictions - targets) ** 2).sum()
print("\n\nRMSE : \n")
rmsTemp=(rmse(np.array(df1_interpolate["temperature"]), np.array(df_file2["temperature"]))/df_file1["temperature"].isnull().sum())**0.5
rmsHum=(rmse(np.array(df1_interpolate["humidity"]), np.array(df_file2["humidity"]))/df_file1["humidity"].isnull().sum())**0.5
rmsPres=(rmse(np.array(df1_interpolate["pressure"]), np.array(df_file2["pressure"]))/df_file1["pressure"].isnull().sum())**0.5
rmsRain=(rmse(np.array(df1_interpolate["rain"]), np.array(df_file2["rain"]))/df_file1["rain"].isnull().sum())**0.5
rmsLavg=(rmse(np.array(df1_interpolate["lightavgw/o0"]), np.array(df_file2["lightavgw/o0"]))/df_file1["lightavgw/o0"].isnull().sum())**0.5
rmsLmax=(rmse(np.array(df1_interpolate["lightmax"]), np.array(df_file2["lightmax"]))/df_file1["lightmax"].isnull().sum())**0.5
rmsMois=(rmse(np.array(df1_interpolate["moisture"]), np.array(df_file2["moisture"]))/df_file1["moisture"].isnull().sum())**0.5

plt.figure(figsize=(6,6))
plt.bar(column1,[rmsTemp,rmsHum,rmsPres,rmsRain,rmsLavg,rmsLmax,rmsMois]);
plt.xticks(rotation=25)
plt.show()

# Q5

for i in ["temperature","rain"] :
    q3=df1_interpolate[i].quantile(.75);
    q1=df1_interpolate[i].quantile(.25);
    iqr=q3-q1
    print("IQR for ",i,": ",iqr)
    min=q1-iqr*1.5
    max=q3+iqr*1.5
    print("\n\nFor : ",i)
    print(df1_interpolate[i][df1_interpolate[i]<=min])
    print(df1_interpolate[i][df1_interpolate[i]>=max])
    df1_interpolate.boxplot(column=[i])
    plt.show()
for i in ["temperature","rain"] :
    med = df1_interpolate[i].median()
    q3=df1_interpolate[i].quantile(.75);
    q1=df1_interpolate[i].quantile(.25);
    iqr=q3-q1
    df1_interpolate[i]=np.where(df1_interpolate[i]<q1-iqr*1.5,med,df1_interpolate[i])
    df1_interpolate[i]=np.where(df1_interpolate[i]>q3+iqr*1.5,med,df1_interpolate[i])
    plt.boxplot(df1_interpolate[i])
    plt.show()
for i in ["temperature","rain"] :
    q3=df1_interpolate[i].quantile(.75);
    q1=df1_interpolate[i].quantile(.25);
    min=q1-iqr*1.5
    max=q3+iqr*1.5
    print("\n\nFor : ",i)
    print(df1_interpolate[i][df1_interpolate[i]<=min])
    print(df1_interpolate[i][df1_interpolate[i]>=max])
    iqr=q3-q1
    print("IQR for ",i,": ",iqr)
