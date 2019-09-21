import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import random

df = pd.read_csv(r'E:\DataFiles\creditcard.csv', low_memory=False)
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())

fraud = df.loc[df['Class'] == 1]
non_fraud = df.loc[df['Class'] == 0]
print(len(fraud))
print(len(non_fraud))

ax = fraud.plot.line(x='Amount', y='Class', color='Red', label='Fraud')
non_fraud.plot.line(x='Amount', y='Class', color='Green', label='Normal', ax=ax)
plt.show()

# Machine Learning
from sklearn import linear_model
from sklearn.model_selection import train_test_split

x = df.iloc[:, :-1]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40)

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X_train, y_train)
print('Score:', logistic.score(X_test, y_test))

y_predicted = np.array(logistic.predict(X_test))
print(y_predicted)
