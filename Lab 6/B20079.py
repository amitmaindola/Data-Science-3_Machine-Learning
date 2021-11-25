'''Name: Amit Maindola
Branch: Computer Science & Engineering
Contact: +91 7470985613'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg as AR

##----------------------------------------------- Question 1 ------------------------------------------
# Question 1 Part (a)
df = pd.read_csv("daily_covid_cases.csv")
plt.figure(figsize=(20,16))
x = ['Feb 20','Apr 20','Jun 20','Aug 20','Oct 20','Dec 20','Feb 21','Apr 21','Jun 21','Aug 21','Oct 21']
plt.xticks([i for i in range(int(612/11), 612 , int(612/11))],x)
plt.plot(df['new_cases'])
plt.show()


#  Question 1 Part (b)
df1 = df['new_cases'][1:]
df2 = df['new_cases'][:-1]

print("Autocorrelation coefficient (for one day lag sequence) = ",pearsonr(list(df1), list(df2))[0])

# Question 1 part (c)
plt.scatter(df1, df2)
plt.show()

# Question 1 Part (d)
corr_cf=[]
lag_value=[]
for i in range(1,7,1):
    df_lag = df['new_cases'][i:]
    df_true = df['new_cases'][:-i]
    corr_cf.append(pearsonr(list(df_lag), list(df_true))[0])
    lag_value.append(i)
plt.plot(lag_value, corr_cf)
plt.show()

# Question 1 Part (e)
plot_acf(df['new_cases'], lags = 6)
plt.show()


##----------------------------------------------- Question 2 ------------------------------------------
# Question 2 Part (a)

series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
X = series.values
test_size = math.ceil(len(X)*0.35)
train, test = X[:len(X)-test_size], X[len(X)-test_size:]
P = 5 # The lag=5
model = AR(train, lags=P) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print("Question-2\nCoefficients found in AR Model\n",coef)
history = train[len(train)-P:]
# print(history)
history = [history[i] for i in range(len(history))]
# print(history)
predictions = list() # List to hold the predictions, 1 step at a time
length = len(history)
for t in range(len(test)):
    lag = [history[i] for i in range(length-P,length)]
    eqn = coef[0] # Initialize to w0
    for d in range(P):
        eqn += coef[d+1] * lag[P-d-1] # Add other values
    obs = test[t]
    predictions.append(eqn) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

# Question 2 Part (b) (i)

plt.scatter(test,predictions)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.show()

# Question 2 Part (b) (ii)

plt.plot(predictions , label="Predicted" )
plt.plot(test , label = "actual")
plt.show()

# Question 2 Part (b) (iii)

n=len(test)
s=0

for i in range(n):
    s=s+(predictions[i]-test[i])**2
avg=sum(test)/len(test)
rmse=(math.sqrt(s/len(test))/avg)*100
print("RMSE between Actual & Predicted: ",rmse)

# For MAPE :
s=0
for i in range(n):
    s=s+ abs(predictions[i]-test[i])/test[i]
mape=(s/n)*100
print("MAPE between Actual & Predicted: ",mape)



##----------------------------------------------- Question 3 ------------------------------------------

def auto_reg(p):
        model = AR(train, lags=p)
        model_fit = model.fit() # fit/train the model
        coef = model_fit.params # Get the coefficients of AR model
        history = train[len(train)-p:]
        history = [history[i] for i in range(len(history))]
        predictions = list() # List to hold the predictions, 1 step at a time
        for t in range(len(test)):
            length = len(history)
            lag = [history[i] for i in range(length-p,length)]
            eqn = coef[0] # Initialize to w0
            for d in range(p):
                eqn += coef[d+1] * lag[p-d-1] # Add other values
            obs = test[t]
            predictions.append(eqn) #Append predictions to compute RMSE later
            history.append(obs) # Append actual test value to history, to be used in next step.
        ###  RMSE Calculation
        n=len(test)
        s=0
        for i in range(n):
            s=s+(predictions[i]-test[i])**2
        avg=sum(test)/len(test)
        rmse=(math.sqrt(s/len(test))/avg)*100
        ### MAPE Calculation
        s=0
        for i in range(n):
            s=s+ abs(predictions[i]-test[i])/test[i]
        mape=(s/n)*100
        #  Returning RMSE and MAPE values
        return rmse[0],mape[0]

rmse=[]
mape=[]
p = []
# Finding RMSE and MPSE
for i in (1,5,10,15,25):
    r , m = auto_reg(i)
    rmse.append(r)
    mape.append(m)
    p.append(i)

# Plotting RMSE with p
plt.bar(p,rmse)
plt.xticks(p)
plt.xlabel("Lag-value")
plt.ylabel("RMSE(%)")
plt.show()
# Plotting MPSE with p
plt.bar(p,mape)
plt.xticks(p)
plt.xlabel("Lag-value")
plt.ylabel("MAPE")
plt.show()


##----------------------------------------------- Question 4 ------------------------------------------
p=1
proceed=True
while(proceed): # To find max value of p
    new_train=train[p:]
    l=len(new_train)
    lag_new_train=train[:l]
    nt =[]
    lnt =[]
    for i in range (len(new_train)):
        nt.append(new_train[i][0])
        lnt. append(lag_new_train[i][0])
    corr = pearsonr(lnt,nt)
    if(2/math.sqrt(l)>abs(corr[0])):
        proceed=False
    else:
        p=p+1

print("Max Value of P value: ",p-1)
p = p-1
model = AR(train, lags=p)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
history = train[len(train)-p:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-p,length)]
    eqn = coef[0] # Initialize to w0
    for d in range(p):
        eqn += coef[d+1] * lag[p-d-1] # Add other values
    obs = test[t]
    predictions.append(eqn) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

n=len(test)

print("For Question 4")
s=0
for i in range(n):
    s=s+(predictions[i]-test[i])**2
avg=sum(test)/len(test)
rmse=(math.sqrt(s/len(test))/avg)*100
print("RMSE between Actual & Predicted: ",rmse)

s=0
for i in range(n):
    s=s+ abs(predictions[i]-test[i])/test[i]
mape=(s/n)*100
print("MAPE between Actual & Predicted: ",mape)