import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def prepare_num(df):
    df_num = df.drop(["Sex", "Embarked", "Pclass"], axis=1)
    df_sex = pd.get_dummies(df["Sex"])
    df_emb = pd.get_dummies(df["Embarked"], prefix="Emb")
    df_pcl = pd.get_dummies(df["Pclass"], prefix="Pclass")
    df_num = pd.concat((df_num, df_sex, df_emb, df_pcl), axis=1)
    return df_num


# Считываем
df = pd.read_csv('data/titanic/train.csv')

# Выбрасываем бесполезные данные и преобразовываем некоторые оставшиеся
df_y = df["Survived"]
df_prep = df.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
df_num = prepare_num(df_prep)
df_num = df_num.fillna(df_num.median())

# Нормализуем данные
scaler = MinMaxScaler()
df_num = scaler.fit_transform(df_num)

# Разделяем данные на train, test, valid
df_train, df_test, df_y_train, df_y_test = train_test_split(df_num, df_y, test_size=0.2, random_state=4)
df_train, df_val, df_y_train, df_y_val = train_test_split(df_train, df_y_train, test_size=0.1, random_state=2)

# Сетка-словарик гиперпараметров
grid_dict = {
    'n_estimators': [x for x in range(10, 50, 10)],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [x for x in range(10, 50, 10)],
    'min_samples_split': [x for x in range(1, 5)],
    'min_samples_leaf': [x for x in range(1, 5)],
    'max_features': ['sqrt', 'log2', None]
}

# Валидационная модель
val_model_core = RandomForestClassifier()
val_model_seek = GridSearchCV(estimator=val_model_core, param_grid=grid_dict, n_jobs=-1, verbose=2)
val_model_seek.fit(df_val, df_y_val)

# Полученные гиперпараметры
params = val_model_seek.best_params_

# Создаем и тренируем модель на основе полученных гиперпараметров
model = RandomForestClassifier(n_estimators=params['n_estimators'],
                               criterion=params['criterion'],
                               max_depth=params['max_depth'],
                               min_samples_split=params['min_samples_split'],
                               min_samples_leaf=params['min_samples_leaf'],
                               max_features=params['max_features'])
model.fit(df_train, df_y_train)

# Проверяем на test и выводим точность
predict = model.predict(df_test)
print(accuracy_score(predict, df_y_test.array))
