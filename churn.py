import joblib

joblib.dump(lr, 'logistic_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')
You said:
import pandas as pd
df= pd.read_csv("Customer churn.csv")
df.head()                                                                                                                                                                   df.info()                                                                                                                                                                      df.describe()                                                                                                                                                                df.isnull().sum()                                                                                                                                                             df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)                                                                                                                 import matplotlib.pyplot as plt
%matplotlib inline
df['Churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()                                                                                                                                                              from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)                                          from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))                                                                                                    from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))                                                                                                                          print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))