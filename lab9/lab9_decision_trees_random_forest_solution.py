import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

loans = pd.read_csv('loan_data.csv')

# EDA
print(loans.info())
print(loans.head())
print(loans.describe())

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,bins=30,label='credit.policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,bins=30,label='credit.policy=0')
plt.legend(); plt.xlabel('FICO'); plt.show()

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,bins=30,label='not.fully.paid=0')
plt.legend(); plt.xlabel('FICO'); plt.show()

plt.figure(figsize=(11,6))
sns.countplot(data=loans, x='purpose', hue='not.fully.paid')
plt.xticks(rotation=45)
plt.show()

sns.jointplot(data=loans, x='fico', y='int.rate', height=8)
plt.show()

sns.lmplot(data=loans, x='fico', y='int.rate', col='not.fully.paid', hue='credit.policy', palette='Set1', height=6, aspect=1)
plt.show()

# Data prep
cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Decision Tree
dtree = DecisionTreeClassifier(random_state=101)
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print('Decision Tree Classification Report:')
print(classification_report(y_test, predictions))
print('Decision Tree Confusion Matrix:')
print(confusion_matrix(y_test, predictions))

# Random Forest
rfc = RandomForestClassifier(n_estimators=600, random_state=101)
rfc.fit(X_train, y_train)
predictions_rf = rfc.predict(X_test)
print('Random Forest Classification Report:')
print(classification_report(y_test, predictions_rf))
print('Random Forest Confusion Matrix:')
print(confusion_matrix(y_test, predictions_rf))
