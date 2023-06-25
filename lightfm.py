from lightfm import LightFM
from lightfm.data import Dataset
import pandas as pd
import numpy as np

# Загрузка данных о прослушиваниях музыки
listening_data = pd.read_csv('listening_history.csv')

# Создание объекта Dataset из данных о прослушиваниях
dataset = Dataset()
dataset.fit(listening_data['user_id'], listening_data['artist_id'])

# Создание разреженной матрицы пользователь-артист на основе данных о прослушиваниях
(interactions, _) = dataset.build_interactions([(user, artist, 1) for user, artist in zip(listening_data['user_id'], listening_data['artist_id'])])

# Использование модели LightFM для построения модели коллаборативной фильтрации
model = LightFM(loss='warp')
model.fit(interactions, epochs=10)

# Функция для получения рекомендаций артистов для пользователя
def get_recommendations(user_id, model, dataset, n=10):
    user_index = dataset.mapping()[0][user_id]
    scores = model.predict(user_index, np.arange(dataset.model_shape[1]))
    top_artist_indices = np.argsort(-scores)[:n]
    recommended_artists = dataset.mapping()[2][top_artist_indices]
    return recommended_artists

# Пример использования
user_id = 1
recommendations = get_recommendations(user_id, model, dataset)
print(f"Рекомендации для пользователя с ID {user_id}:")
print(recommendations)
