import operator
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

# Определяем важные признаки
features = df.columns
importances = model.feature_importances_
importances_dict = dict(zip(features, importances))
importances_dict_sorted = sorted(importances_dict.items(), key=operator.itemgetter(1), reverse=True)[0:2]
important_features = []
for feature, importance in importances_dict_sorted:
    important_features.append(feature)

# Редактируем данные с учетом полученных признаков
df_new = df.loc[:, important_features]
df_new_scaled = scaler.fit_transform(df_new)

df_new_train, df_new_test, df_new_train_y, df_new_test_y = train_test_split(df_new_scaled,
                                                                            df_y,
                                                                            test_size=0.2,
                                                                            random_state=2)
model_new = XGBClassifier()
model_new.fit(df_new_train, df_new_train_y)

predict_new = model_new.predict(df_new_test)
acc_new = accuracy_score(df_new_test_y, predict_new)

print('Точность (на основании двух важнейших): ', end='')
print(acc_new)
