import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("insurance_data.csv")
print(df.head())
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
#plt.show()

print(df.shape)

x_train, x_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.1)
print(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

print(f"x_test: {x_test}")
predict = model.predict(x_test)
print(f"predict: {predict}")
# predict_value = '22'
# print(f"Age {predict_value}, Predict bought insurance: {model.predict([[predict_value]])}")
