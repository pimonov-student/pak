import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

scaler = MinMaxScaler()

# Читаем
df = pd.read_csv('titanic_prepared.csv')
df_y = df['label']
df = df.drop('label', axis=1)
df_scaled = scaler.fit_transform(df)

# Разделяем
df_train, df_test, df_train_y, df_test_y = train_test_split(df_scaled, df_y, test_size=0.2, random_state=2)

# Обучаем
model = XGBClassifier()
model.fit(df_train, df_train_y)

predict = model.predict(df_test)
acc = accuracy_score(df_test_y, predict)

print('Точность: ', end='')
print(acc)
