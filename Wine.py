import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')
dtc = DecisionTreeClassifier()

grade = []
for i in data['quality']:
    if i >=7:
        grade.append(1)
    else:
        grade.append(0)

data['quality'] = grade

x = data.drop(["type", "residual sugar", "pH", "volatile acidity", "quality"], axis=1)
y = data['quality']

x.fillna(x.mean(), inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
print("Accuracy Score: ",accuracy_score(y_test, y_pred))

'''
Accuracy Score:  0.8523076923076923
'''