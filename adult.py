import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cluster import DBSCAN



print(os.getcwd())
df = pd.read_csv("./data/adult.csv")
print(df["workclass"].value_counts())
df=df.replace('?', None)
print(df["workclass"].value_counts())
print(df.head())
print(df.isna().sum())
print(df.shape)
df=df.dropna(how="any")
print(df.shape)
print(df.columns)
print(df["race"])
df=df.drop(columns=["fnlwgt","marital.status","education","relationship"])
print(df.shape)
print(df.columns)


#les colonnes problématiques restantes sont: (workclass ,race sex,native_country,occupation )


#sex
df['sex'].replace(['Female','Male'],[0,1],inplace=True)
print(df["sex"])

#workclass
print(df["workclass"].value_counts())
df['workclass'].replace(['Private','Self-emp-not-inc',"Local-gov","State-gov","Self-emp-inc","Federal-gov","Without-pay"],[0,1,2,3,4,5,6],inplace=True)
print(df["workclass"].value_counts())

#race
print(df["race"].value_counts())
df['race'].replace(['White','Black',"Asian-Pac-Islander","Amer-Indian-Eskimo","Other"],list(range(0,5)),inplace=True)
print(df["race"].value_counts())

#native.country
#print(len(df["native.country"].value_counts()))
#print(list(set(df["native.country"].tolist())))
values= list(set(df["native.country"].tolist()))
df['native.country'].replace(values,list(range(0,len(values))),inplace=True)
print(df["native.country"].value_counts())

#occupation
print(df["occupation"].value_counts())
values= list(set(df["occupation"].tolist()))
df['occupation'].replace(values,list(range(0,len(values))),inplace=True)
print(df["occupation"].value_counts())
#print the column type to ckeck that we don't have anymore string value
print(df.dtypes)
#voir les catégories de income et les traiter pour les rendre numériques
print(df["income"].value_counts())
values= list(set(df["income"].tolist()))
df['income'].replace(values,list(range(0,len(values))),inplace=True)
print(df["income"].value_counts())
#print the column type to ckeck that we don't have anymore string value
print(df.dtypes)

X=df.drop(columns="income")
y=df["income"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.7, random_state=42)
clf= KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
knn_score = clf.score(X_test,y_test)
print(knn_score)

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))

X=df.drop(columns=["income","native.country","race","occupation"])
y=df["income"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.7, random_state=42)
clf= KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
knn_score = clf.score(X_test,y_test)
print("the new score is:",knn_score)