import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import data_preprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_generation', action='store_true')
args = parser.parse_args()

class FeatureGenerator:
    def __init__(self, n_clusters=5, n_neighbors=3, poly_degree=2):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.poly_degree = poly_degree
        self.cat_features = []
        self.num_features = []
        self.time_features = []
        self.groupby_aggs = {}
        
    def detect_features(self, df):
        """Определение типов признаков"""
        self.cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_features = df.select_dtypes(include=np.number).columns.tolist()
        
        # Поиск временных признаков
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64) or 'date' in col.lower() or 'time' in col.lower():
                self.time_features.append(col)
                
    # def handle_categorical(self, df, target=None):
    #     """Обработка категориальных признаков"""
    #     # One-Hot для признаков с малым числом уникальных значений
    #     for col in self.cat_features:
    #         if df[col].nunique() < 10:
    #             dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    #             df = pd.concat([df, dummies], axis=1)
        
    #     # Target Encoding для признаков с большим числом уникальных значений
    #     if target:
    #         for col in self.cat_features:
    #             if df[col].nunique() >= 10:
    #                 df[f'{col}_target_enc'] = self.target_encoding(df, col, target)
    #     return df
    
    # def target_encoding(self, df, col, target):
    #     """Кросс-валидационное Target Encoding"""
    #     enc = pd.Series(np.zeros(len(df)), index=df.index)
    #     kf = KFold(n_splits=5, shuffle=True)
        
    #     for trn_idx, val_idx in kf.split(df):
    #         trn = df.iloc[trn_idx]
    #         means = trn.groupby(col)[target].mean()
    #         enc.iloc[val_idx] = df.iloc[val_idx][col].map(means)
        
    #     return enc.fillna(df[target].mean())
    
    def generate_features(self, df, target=None):
        """Основной метод генерации признаков"""
        self.detect_features(df)
        self.target = target
                
        '''
        В данной функции у нас обрабатываются категориальные признаки, по-сути идет предобработка данных, 
        в каждом отдельном датасете, как будето это стоит делать отдельно. 
        Заполнять пропуски, кодировать категориальные признаки и т д
        '''
        #! df = self.handle_categorical(df, target)
    
        # Удаление исходных категориальных признаков
        #! df = df.drop(columns=self.cat_features)
        
        # Обновление списка числовых признаков
        self.num_features = df.select_dtypes(include=np.number).columns.tolist()
        
        # Убираем из списка признаков target колонку
        if self.target is not None:
            self.num_features.remove(target)
        
        # Генерация новых признаков
        df = self.polynomial_features(df)
        df = self.transform_features(df)
        df = self.interaction_features(df)
        df = self.temporal_features(df)
        df = self.cluster_features(df)
        # df = self.neighbor_features(df) #! пока не понятно, как это обрабатывать, см коммент к этой функции
        
        return df
    
    def polynomial_features(self, df):
        #  Выделяем target столбец. Он не будет участвовать в математических преобразованиях
        if self.target is not None:
            target_column = df[self.target]
        else:
            target_column = None
        
        """Полиномиальные признаки"""
        poly = PolynomialFeatures(degree=self.poly_degree, 
                                 interaction_only=False, 
                                 include_bias=False)
        poly_features = poly.fit_transform(df[self.num_features])
        poly_cols = poly.get_feature_names_out(self.num_features)
        poly_df = pd.DataFrame(poly_features, columns=poly_cols, index=df.index)
        
        if target_column is not None:
            result_df = pd.concat([target_column, poly_df], axis=1)
        
        return result_df
    
    def transform_features(self, df):
        """Математические трансформации"""
        new_features = []
        for col in tqdm(self.num_features, desc='Feature transformation'):
            
            if df[col].min().item() > 0:  # Для логарифма нужны положительные значения
                df[f'log_{col}'] = np.log1p(df[col])
                new_features.append(f'log_{col}')
            df[f'sqrt_{col}'] = np.sqrt(np.abs(df[col]))
            df[f'sqr_{col}'] = df[col]**2
            df[f'recip_{col}'] = 1 / (df[col] + 1e-5)
            new_features.extend([f'sqrt_{col}', f'sqr_{col}', f'recip_{col}'])
        
        # Обновление списка числовых признаков
        self.num_features.extend(new_features)
        return df
    
    def interaction_features(self, df):
        """Попарные взаимодействия признаков"""
        n = len(self.num_features)
        for i in tqdm(range(n), desc='Generating interactions'):
            for j in range(i+1, n):
                col1, col2 = self.num_features[i], self.num_features[j]
                df[f'mul_{col1}_{col2}'] = df[col1] * df[col2]
                df[f'div_{col1}_{col2}'] = df[col1] / (df[col2] + 1e-5)
                df[f'sum_{col1}_{col2}'] = df[col1] + df[col2]
                df[f'diff_{col1}_{col2}'] = df[col1] - df[col2]
        return df
    
    def temporal_features(self, df):
        """Признаки из временных данных"""
        for col in tqdm(self.time_features, desc='Generating temporal features'):
            dt_col = pd.to_datetime(df[col])
            df[f'{col}_year'] = dt_col.dt.year
            df[f'{col}_month'] = dt_col.dt.month
            df[f'{col}_day'] = dt_col.dt.day
            df[f'{col}_dow'] = dt_col.dt.dayofweek
            df[f'{col}_hour'] = dt_col.dt.hour
            df[f'{col}_quarter'] = dt_col.dt.quarter
            df[f'{col}_is_weekend'] = df[f'{col}_dow'] >= 5
            
            # Разности во времени
            if df.index.is_monotonic_increasing:
                df[f'{col}_diff_prev'] = df[col].diff()
        return df
    
    def cluster_features(self, df):
        """Признаки кластеризации"""
        if len(self.num_features) > 1:
            kmeans = KMeans(n_clusters=self.n_clusters)
            df['cluster'] = kmeans.fit_predict(df[self.num_features])
        return df
    
    '''
    Данная функция для каждой строки датафрейма генерируют сборную соляночку.
    В каждой строке получается не одно значение, а кортеж из устредненных значений по всем
    колонкам оригинального датасета. Поэтому нужно подумать, как это обрабатывать.
    
    Видимо, нужно разбивать их по колонкам и давать индивидуальные названия.
    '''
    def neighbor_features(self, df):
        """Признаки на основе соседей"""
        if len(self.num_features) > 1:
            nn = NearestNeighbors(n_neighbors=self.n_neighbors)
            nn.fit(df[self.num_features])
            distances, indices = nn.kneighbors()
            
            for stat in ['mean', 'max', 'min']:
                df[f'neighbor_{stat}'] = [getattr(df.iloc[i][self.num_features], stat)() 
                                         for i in indices]
        return df

class FeatureSelector:
    def __init__(self, threshold=0.9, importance_threshold=0.001):
        self.threshold = threshold
        self.importance_threshold = importance_threshold
        self.selected_features = []
        
    def variance_selection(self, df):
        """Отбор признаков по дисперсии"""
        selector = VarianceThreshold(threshold=(self.threshold * (1 - self.threshold)))
        selector.fit(df)
        return df.columns[selector.get_support()]
    
    def importance_selection(self, X, y, params=None):
        """Отбор признаков по важности LightGBM"""
        if params is None:
            params = {
                'objective': 'binary' if len(y.unique()) == 2 else 'regression',
                'boosting_type': 'gbdt',
                'n_estimators': 500,
                'importance_type': 'gain',
                'random_state': 42
            }
        
        model = lgb.LGBMClassifier(**params) if len(y.unique()) == 2 else lgb.LGBMRegressor(**params)
        model.fit(X, y)
        
        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance = importance[importance > importance.max() * self.importance_threshold]
        
        return importance.index.tolist()
    
    def select_features(self, df, target):
        """Основной метод отбора признаков"""
        # Удаление константных и квази-константных признаков
        variance_selected = self.variance_selection(df.drop(columns=[target]))
        
        x = variance_selected.tolist() + [target]
        
        df_filtered = df[variance_selected.tolist() + [target]]
        
        # Отбор по важности
        X = df_filtered.drop(columns=[target])
        y = df_filtered[target]
        self.selected_features = self.importance_selection(X, y)
        
        return df[self.selected_features + [target]]

if __name__ == '__main__':
    # Загрузка данных
    df = pd.read_csv('./data/train.csv')
    df = data_preprocessing(df)
    
    target_col = 'Survived'

    # Генерация признаков
    generator = FeatureGenerator(n_clusters=5, n_neighbors=5, poly_degree=2)
    feature_rich_data = generator.generate_features(df, target=target_col)
    
    # Отбор признаков
    selector = FeatureSelector(threshold=0.95, importance_threshold=0.005)
    selected_data = selector.select_features(feature_rich_data, target_col)

    print(f"Исходное число признаков: {df.shape[1]}")
    print(f"Сгенерировано признаков: {feature_rich_data.shape[1]}")
    print(f"Отобрано значимых признаков: {len(selector.selected_features)}")

    # Сохранение результата
    selected_data.to_csv('selected_features_data.csv', index=False)
    selected_data_columns = selected_data.columns.to_list()
    selected_data_columns.remove(target_col)
        
    if args.test_generation:
        test_df = pd.read_csv('./data/test.csv')
        test_df = data_preprocessing(test_df)
        
        feature_rich_data_test = generator.generate_features(test_df, target=None)
        feature_rich_data_test = feature_rich_data_test[selected_data_columns]
        feature_rich_data_test.to_csv('feature_rich_data_test.csv')