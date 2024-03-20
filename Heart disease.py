# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:43:18 2024

@author: Dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
     
df = pd.read_csv('C:/Users/Dell/Desktop/Heart Disease data/Heart Disease data.csv')
df.head()

df.shape

df.keys()

df.info()     

df.describe()

df.isna().sum()

df['target'].value_counts()

plt.figure(figsize = (10, 10)) 
sns.heatmap(df.corr(), cmap='Purples',annot=True, linecolor='Green', linewidths=1.0)
plt.show()

sns.pairplot(df)
plt.show()

sns.catplot(data=df, kind='count', x='age',hue='sex',aspect=2)
plt.show()

sns.catplot(data=df, kind='count', x='target', col='sex',row='age', palette='Blues')
plt.show()

X = df.iloc[:,0:13]
y = df.iloc[:,13:14]

X.head()

y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

score = logreg.score(X_test, y_test)
print("Prediction score is:",score)      

from sklearn.metrics import confusion_matrix, classification_report 
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix is:\n",cm)

print("Classification Report is:\n\n",classification_report(y_test,y_pred))

conf_matrix = pd.DataFrame(data = cm,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (10, 6)) 
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens", linecolor="Blue", linewidths=1.5) 
plt.show() 