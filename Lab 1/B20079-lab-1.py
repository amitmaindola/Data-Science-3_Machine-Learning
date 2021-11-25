# Name : Amit Maindola
# Registration N0. : B20079
# Mobile NO. : +91 7470985613
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("pima-indians-diabetes.csv")

# Part 1

# Mean
print("\n\nMean of all the columns is given below : ")
print(df[['pregs','plas','pres','skin','test','BMI','pedi','Age']].mean(axis=0), "\n")

# Median
print("Median of all the columns is given below : ")
print(df[['pregs','plas','pres','skin','test','BMI','pedi','Age']].median(axis=0), "\n")

# Mode
print("Mode of all the columns is given below : ")
modes=df[['pregs','plas','pres','skin','test','BMI','pedi','Age']].mode(axis=0)
print((modes.iloc[1].fillna(modes.iloc[0]) + modes.iloc[0])/2)

# Min
print("Minimum values of all the columns is given below : ")
print(df[['pregs','plas','pres','skin','test','BMI','pedi','Age']].min(axis=0), "\n")

# Max
print("Maximum values of all the columns is given below : ")
print(df[['pregs','plas','pres','skin','test','BMI','pedi','Age']].max(axis=0), "\n")

# Standard deviation
print("Standard deviations of all the columns is given below : ")
print(df[['pregs','plas','pres','skin','test','BMI','pedi','Age']].std(axis=0), "\n")


# Part 2

# Age vs press
plt.scatter(df["Age"],df["pres"],s=20, color="b", marker="o", alpha=0.4)
plt.title("Age vs Diastolic blood pressure")
plt.xlabel(" Age (years) ")
plt.ylabel("Diastolic blood pressure (mm Hg) ")
plt.show()

# Age vs pregs
plt.scatter(df["Age"],df["pregs"],s=20, color="b", marker="o", alpha=0.4)
plt.title("Age vs Number of times pregnant")
plt.xlabel(" Age (years) ")
plt.ylabel("Number of times pregnant")
plt.show()

# Age vs plas
plt.scatter(df["Age"],df["plas"],s=20, color="b", marker="o", alpha=0.4)
plt.title("Age vs Plas")
plt.xlabel(" Age (years) ")
plt.ylabel("Plas")
plt.show()

# Age vs skin
plt.scatter(df["Age"],df["skin"],s=20, color="b", marker="o", alpha=0.4)
plt.title("Age vs Skin")
plt.xlabel(" Age (years) ")
plt.ylabel("Skin")
plt.show()

# Age vs BMI
plt.scatter(df["Age"],df["BMI"],s=20, color="b", marker="o", alpha=0.4)
plt.title("Age vs BMI")
plt.xlabel(" Age (years) ")
plt.ylabel("BMI")
plt.show()

# Age vs test
plt.scatter(df["Age"],df["test"],s=20, color="b", marker="o", alpha=0.4)
plt.title("Age vs Test")
plt.xlabel(" Age (years) ")
plt.ylabel("Test")
plt.show()

# Age vs pedi
plt.scatter(df["Age"],df["pedi"],s=20, color="b", marker="o", alpha=0.4)
plt.title("Age vs Diabetes pedigree function ")
plt.xlabel(" Age (years) ")
plt.ylabel("Diabetes pedigree function ")
plt.show()



# BMI vs pres
plt.scatter(df["BMI"],df["pres"],s=20, color="b", marker="o", alpha=0.4)
plt.title("BMI vs Diastolic blood pressure")
plt.xlabel(" BMI ")
plt.ylabel("Diastolic blood pressure (mm Hg) ")
plt.show()

# BMI vs pregs
plt.scatter(df["BMI"],df["pregs"],s=20, color="b", marker="o", alpha=0.4)
plt.title("BMI vs Number of times pregnant")
plt.xlabel(" BMI ")
plt.ylabel("Number of times pregnant")
plt.show()

# BMI vs plas
plt.scatter(df["BMI"],df["plas"],s=20, color="b", marker="o", alpha=0.4)
plt.title("BMI vs Plas")
plt.xlabel(" BMI ")
plt.ylabel("Plas")
plt.show()

# BMI vs skin
plt.scatter(df["BMI"],df["skin"],s=20, color="b", marker="o", alpha=0.4)
plt.title("BMI vs Skin")
plt.xlabel(" BMI ")
plt.ylabel("Skin")
plt.show()

# BMI vs Age
plt.scatter(df["BMI"],df["Age"],s=20, color="b", marker="o", alpha=0.4)
plt.title("BMI vs Age")
plt.xlabel(" BMI ")
plt.ylabel("Age (years)")
plt.show()

# BMI vs test
plt.scatter(df["BMI"],df["test"],s=20, color="b", marker="o", alpha=0.4)
plt.title("BMI vs Test")
plt.xlabel(" BMI ")
plt.ylabel("Test")
plt.show()

# BMI vs pedi
plt.scatter(df["BMI"],df["pedi"],s=20, color="b", marker="o", alpha=0.4)
plt.title("BMI vs Diabetes pedigree function ")
plt.xlabel(" BMI ")
plt.ylabel("Diabetes pedigree function ")
plt.show()




# Part 3

# With Age

print("\ncorrelation coefficient of Age with Pregs")
print(np.corrcoef(df["Age"],df["pregs"])[0][1])

print("\ncorrelation coefficient of Age with Plas")
print(np.corrcoef(df["Age"],df["plas"])[0][1])

print("\ncorrelation coefficient of Age with Pres")
print(np.corrcoef(df["Age"],df["pres"])[0][1])

print("\ncorrelation coefficient of Age with Skin")
print(np.corrcoef(df["Age"],df["skin"])[0][1])

print("\ncorrelation coefficient of Age with Test")
print(np.corrcoef(df["Age"],df["test"])[0][1])

print("\ncorrelation coefficient of Age with BMI")
print(np.corrcoef(df["Age"],df["BMI"])[0][1])

print("\ncorrelation coefficient of Age with Pedi")
print(np.corrcoef(df["Age"],df["pedi"])[0][1])

# With BMI
print("\ncorrelation coefficient of BMI with Pregs")
print(np.corrcoef(df["BMI"],df["pregs"])[0][1])

print("\ncorrelation coefficient of BMI with Plas")
print(np.corrcoef(df["BMI"],df["plas"])[0][1])

print("\ncorrelation coefficient of BMI with Pres")
print(np.corrcoef(df["BMI"],df["pres"])[0][1])

print("\ncorrelation coefficient of BMI with Skin")
print(np.corrcoef(df["BMI"],df["skin"])[0][1])

print("\ncorrelation coefficient of BMI with Test")
print(np.corrcoef(df["BMI"],df["test"])[0][1])

print("\ncorrelation coefficient of BMI with Age")
print(np.corrcoef(df["BMI"],df["Age"])[0][1])

print("\ncorrelation coefficient of BMI with Pedi")
print(np.corrcoef(df["BMI"],df["pedi"])[0][1])



# Part 4

plt.hist(df["pregs"])
plt.title("Histogram for pregs")
plt.show()

plt.hist(df["skin"])
plt.title("Histogram for Skin")
plt.show()



# Part 5

df['pregs'].hist(by=df["class"],figsize=[15,5])
plt.show()




# Part 6

plt.boxplot(df["pregs"])
plt.show()

plt.boxplot(df["plas"])
plt.show()

plt.boxplot(df["pres"])
plt.show()

plt.boxplot(df["skin"])
plt.show()

plt.boxplot(df["test"])
plt.show()

plt.boxplot(df["BMI"])
plt.show()

plt.boxplot(df["pedi"])
plt.show()

plt.boxplot(df["Age"])
plt.show()