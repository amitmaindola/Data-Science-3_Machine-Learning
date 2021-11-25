import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv("SteelPlateFaults-2class.csv")

x_label=df["Class"]

# Question 1
print("Question 1\n")
data_class_0=df[df["Class"]==0]
data_class_1=df[df["Class"]==1]
x_label_0=data_class_0["Class"]
x_label_1=data_class_1["Class"]
# Spilting data for both classes saperately
x_train_0, x_test_0, x_label_train_0, x_label_test_0  = train_test_split(data_class_0, x_label_0, test_size=0.3, random_state=42, shuffle=True)
x_train_1, x_test_1, x_label_train_1, x_label_test_1  = train_test_split(data_class_1, x_label_1, test_size=0.3, random_state=42, shuffle=True)
# Merging the test data and train data from both the classes
x_train=pd.concat([x_train_0, x_train_1])
x_test=pd.concat([x_test_0, x_test_1])
x_label_train=pd.concat([x_label_train_0, x_label_train_1])
x_label_test=pd.concat([x_label_test_0, x_label_test_1])

# Exporting the test data and train data as csv files
x_train.to_csv('SteelPlateFaults-train.csv')
x_test.to_csv(' SteelPlateFaults-test.csv')

#Question 1)
x_train.drop('Class', axis=1, inplace=True)
x_test.drop('Class', axis=1, inplace=True)
best_result=[]
best=0
for k in range (1,6,2): # For k=1,3,5
    print("For k =", k)
    # Part (a)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, x_label_train)
    x_pred = model.predict(x_test)
    confMatrix=confusion_matrix(x_label_test, x_pred)
    print(confMatrix)
    # Part (b)
    accuracy=(confMatrix[0][0]+confMatrix[1][1])/(confMatrix[0][0]+confMatrix[1][1]+confMatrix[0][1]+confMatrix[1][0])
    print("Accuracy =", accuracy*100, "%")
    if accuracy>best:
        best=accuracy
best_result.append(best)

# Question 2)
print("Question 2\n")
scaler = MinMaxScaler()
scaler.fit(x_train) # Normalisation
x_train_normal = pd.DataFrame(scaler.transform(x_train))
x_train_normal.to_csv('SteelPlateFaults-train-Normalised.csv')

x_test_normal = pd.DataFrame(scaler.transform(x_test))
x_test_normal.to_csv('SteelPlateFaults-test-Normalised.csv')

best=0
for k in range (1,6,2): # For k=1,3,5
    print("For k =", k)
    # Part (a)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_normal, x_label_train)

    x_pred = model.predict(x_test_normal)
    confMatrix=confusion_matrix(x_label_test, x_pred)
    print(confMatrix)
    # Part (b)
    accuracy=(confMatrix[0][0]+confMatrix[1][1])/(confMatrix[0][0]+confMatrix[1][1]+confMatrix[0][1]+confMatrix[1][0])
    print("Accuracy =", accuracy*100, "%")
    if accuracy>best:
        best=accuracy
best_result.append(best)


# Question 3

def likelihood(x_vector,mean_vector,cov_matrix): # defining likelihood function
    matrix1= np.dot((x_vector-mean_vector).T,np.linalg.inv(cov_matrix))
    pow= -0.5*np.dot(matrix1,(x_vector-mean_vector))
    exp_part=np.exp(pow)
    # exp_part=bigfloat.exp(pow,bigfloat.precision(100))
    return (exp_part/((2*np.pi)**11.5 * (abs(np.linalg.det(cov_matrix)))**.5))

# calculating prior for both classes
Ni_0=0
Ni_1=0
Ntotal=0
for i in x_label_train:
    if i==1:
        Ni_1+=1
    else:
        Ni_0+=1
    Ntotal+=1
prior_0=Ni_0/Ntotal
prior_1=Ni_1/Ntotal

# calculating mean vector and covariance matrix
x_train_0.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], inplace=True)
x_train_1.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], inplace=True)
mean_0=np.array(x_train_0[x_train_0.columns[:-1]].mean()) # Just removing the attribute 'Class' and then creating a mean vector
mean_1=np.array(x_train_1[x_train_1.columns[:-1]].mean())
cov_0=np.cov((x_train_0[x_train_0.columns[:-1]]).T)
cov_1=np.cov((x_train_1[x_train_1.columns[:-1]]).T)

bayes_prediction=[] # To store predictions
x_test.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], inplace=True) # Droping some unnecessary attributes
for row in x_test.itertuples(index=False):
        l0=likelihood(np.array(row),mean_0,cov_0)*prior_0
        l1=likelihood(np.array(row),mean_1,cov_1)*prior_1
        if l0>l1:
            bayes_prediction.append(0)
        else:
            bayes_prediction.append(1)
print("\nBayes Classification Predictions:")
confMatrix=confusion_matrix(x_label_test, bayes_prediction)
print(confMatrix)
accuracy=(confMatrix[0][0]+confMatrix[1][1])/(confMatrix[0][0]+confMatrix[1][1]+confMatrix[0][1]+confMatrix[1][0])
print("Accuracy =", accuracy*100, "%")
best_result.append(accuracy)

# Getting csv files for covariance matrix for adding that screenshot in lab report
index_values=[]
column_values=[]
for i in range(1,24):
    index_values.append(i)
    column_values.append(i)
cov_0_df=pd.DataFrame(data=cov_0,index=index_values, columns=column_values)
cov_1_df=pd.DataFrame(data=cov_1,index=index_values, columns=column_values)
cov_0_df.to_excel("Covariance_matrix_Class_0.xlsx")
cov_1_df.to_excel("Covariance_matrix_Class_1.xlsx")


# Question 4
print("\n\nThe Best Results In Different Cases Is")
print("\nBest result of KNN classifier:",best_result[0]*100, "%")
print("\nBest result of KNN classifier on normalised data:",best_result[1]*100, "%")
print("\nBest result using Bayes classifier:",best_result[2]*100, "%")