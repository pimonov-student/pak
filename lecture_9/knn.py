import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
    'algorithm':    ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size':    [x for x in range(1, 20)],
    'n_neighbors':  [x for x in range(1, 20)],
    'p':            [1, 2, 3],
    'weights':      ['uniform', 'distance']
}

# Валидационная модель
val_model_core = KNeighborsClassifier()
val_model_seek = GridSearchCV(estimator=val_model_core, param_grid=grid_dict, n_jobs=-1, verbose=2)
val_model_seek.fit(df_val, df_y_val)

# Полученные гиперпараметры
params = val_model_seek.best_params_

# Создаем и тренируем модель на основе полученных гиперпараметров
model = KNeighborsClassifier(algorithm=params['algorithm'],
                             leaf_size=params['leaf_size'],
                             n_neighbors=params['n_neighbors'],
                             p=params['p'],
                             weights=params['weights'])
model.fit(df_train, df_y_train)

# Проверяем на test и выводим точность
predict = model.predict(df_test)
print(accuracy_score(predict, df_y_test.array))
