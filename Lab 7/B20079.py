# Name: Amit Maindola (B20079)
# Contact: +91 7470985613
# Branch: Computer Science & Engineering

from operator import mod
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

################################################################ Question 1 ################################################################################
print("\nQuestion 1")
df = pd.read_csv('Iris.csv') # Reading the dataset
df_label=df['Species']
x_label = [] # Converting the label class as numeric value
for i in range (len(df_label)):
    if (df_label[i] == 'Iris-setosa'):
        x_label.append(0)
    if (df_label[i] == 'Iris-versicolor'):
        x_label.append(1)
    if (df_label[i] == 'Iris-virginica'):
        x_label.append(2)
before_pca = df.drop(columns='Species')

eigenvalues,eigenvectors= np.linalg.eig(np.cov(before_pca.T)) # Finding Eigenvalues and Eigenvectors
model = PCA(n_components=2) # Building a pca model to reduce data to 2 diamensions
pca_result = model.fit_transform(before_pca) # GEtting results through our pca model
x = before_pca
#plotting
c = np.linspace(1,4,4)
plt.bar(c,[round(i,3) for i in eigenvalues])
plt.xticks(np.arange(min(c), max(c)+1, 1.0))
plt.show()

print("EigenValues: ",eigenvalues)

################################################################ Question 2 ################################################################################
print("\nQuestion 2")
model = KMeans(n_clusters=3) # K Means clustering model
b = model.fit(pca_result)
label = model.fit_predict(pca_result)

for i in range(3): # Plotting for all labels
    c=['r', 'b','y']
    label_i = pca_result[ label == i]
    plt.scatter(label_i[:,0] , label_i[:,1],  color = c[i])
plt.show()

print("Distortion Measure: ",b.inertia_)

def purity_score(y_true, y_pred): # Utility function to find purity_score
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred) #Computing confusion matrix
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix) # Find optimal one-to-one mapping between cluster labels and true labels
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix) # Return cluster accuracy

print("Purity Score:", purity_score(x_label,label))

################################################################ Question 3 ################################################################################
print("\nQuestion 3")
# K Means Clustering for different values of k
distortion_list=[]
p_score =[]
K = [2,3,4,5,6,7]
for i in K:
    model = KMeans(n_clusters=i)
    b = model.fit(pca_result)
    label = model.fit_predict(pca_result)
    distortion_list.append(b.inertia_)
    p_score.append(purity_score(x_label , label))
plt.plot(K,distortion_list) # plotting the graph of distortion measure wrt values of k
plt.show()
print("Purity Scores For k = [2,3,4,5,6,7]: ",p_score)


################################################################ Question 4 ################################################################################
print("\nQuestion 4")
# GMM Clustering for k = 3
k = 3
model = GaussianMixture(n_components = k)
model.fit(pca_result)
label = model.predict(pca_result)

for i in range(3): # Plotting the cluster
    c=['r', 'b','y']
    label_i0 = pca_result[ label == i]
    plt.scatter(label_i0[:,0] , label_i0[:,1],  color = c[i])
plt.show()

att = before_pca.columns
x = model.lower_bound_
print(x*(len(before_pca[att[1]])))
print("Purity Scores: ",purity_score(x_label,label))

################################################################ Question 5 ################################################################################
print("\nQuestion 5")
# GMM Clustering for different values of k
likelihood = []
p_score_gmm =[]
for i in K:
    gmm = GaussianMixture(n_components = i , random_state=5)
    gmm.fit(pca_result)
    label = gmm.predict(pca_result)
    likelihood.append(gmm.lower_bound_ * len(before_pca[att[1]]))
    p_score_gmm.append(purity_score(x_label , label))
plt.plot(K,likelihood)
plt.show()
print("Purity Scores For k = [2,3,4,5,6,7]: ",p_score_gmm)

################################################################ Question 6 ################################################################################
print("\nQuestion 6")
#getting the species in the df frame for dbscan
d = {'Species' : x_label}
m = pd.DataFrame(data=d)
df.Species= m.Species
def DBSCAN_(ep , samples):
    dbscan_model = DBSCAN(eps=ep, min_samples=samples).fit(pca_result)
    return dbscan_model.labels_

eps = [1, 1, 5, 5]
min_samples = [4, 10, 4, 10]
for i in range(4):
    model = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(pca_result)
    DBSCAN_predictions = model.labels_
    print(f'Purity score for eps={eps[i]} and min_samples={min_samples[i]} is', round(purity_score(x_label, DBSCAN_predictions), 3))
    plt.scatter(pca_result[:,0], pca_result[:,1], c=DBSCAN_predictions, cmap='flag', s=15)
    plt.title(f'df Points for eps={eps[i]} and min_samples={min_samples[i]}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()