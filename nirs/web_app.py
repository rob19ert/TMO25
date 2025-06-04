import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Загрузка модели
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Загрузка и подготовка данных (создание df_selected)
df = pd.read_csv('train.csv')
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                     'FullBath', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                     'Fireplaces', 'BsmtFinSF1', 'LotArea', 'WoodDeckSF',
                     'OpenPorchSF', 'Neighborhood', 'HouseStyle', 'SaleCondition']
df_selected = df[selected_features + ['SalePrice']].copy()
df_selected['MasVnrArea'] = df_selected['MasVnrArea'].fillna(0)

st.title('Предсказание цены дома')

st.header('Введите параметры дома')

# Числовые параметры
overall_qual = st.slider('Общее качество (1-10)', 1, 10, 5)
gr_liv_area = st.number_input('Жилая площадь (кв.футы)', min_value=0, value=1500)
garage_cars = st.slider('Размер гаража (кол-во машин)', 0, 4, 2)
total_bsmt_sf = st.number_input('Площадь подвала (кв.футы)', min_value=0, value=1000)
full_bath = st.slider('Количество ванных', 0, 4, 2)
year_built = st.slider('Год постройки', 1800, 2023, 2000)
year_remod_add = st.slider('Год ремонта', 1800, 2023, 2000)
mas_vnr_area = st.number_input('Площадь облицовки (кв.футы)', min_value=0, value=0)
fireplaces = st.slider('Количество каминов', 0, 4, 0)
bsmt_fin_sf1 = st.number_input('Готовая площадь подвала (кв.футы)', min_value=0, value=500)
lot_area = st.number_input('Площадь участка (кв.футы)', min_value=0, value=10000)
wood_deck_sf = st.number_input('Площадь деревянной террасы (кв.футы)', min_value=0, value=0)
open_porch_sf = st.number_input('Площадь открытой веранды (кв.футы)', min_value=0, value=50)

# Категориальные параметры
neighborhood = st.selectbox('Район', df_selected['Neighborhood'].unique())
house_style = st.selectbox('Стиль дома', df_selected['HouseStyle'].unique())
sale_condition = st.selectbox('Условия продажи', df_selected['SaleCondition'].unique())

# Модельные параметры
if st.checkbox('Настроить параметры модели'):
    n_estimators = st.slider('Количество деревьев', 50, 500, 100)
    learning_rate = st.slider('Скорость обучения', 0.01, 0.2, 0.1)
    max_depth = st.slider('Макс. глубина дерева', 1, 10, 3)

    # Обучение новой модели
    numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                        'FullBath', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                        'Fireplaces', 'BsmtFinSF1', 'LotArea', 'WoodDeckSF', 'OpenPorchSF']
    categorical_features = ['Neighborhood', 'HouseStyle', 'SaleCondition']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        ))
    ])

    X = df_selected.drop('SalePrice', axis=1)
    y = df_selected['SalePrice']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    st.success("Модель переобучена с новыми параметрами!")

# Предсказание
if st.button('Предсказать цену'):
    input_data = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'GarageCars': [garage_cars],
        'TotalBsmtSF': [total_bsmt_sf],
        'FullBath': [full_bath],
        'YearBuilt': [year_built],
        'YearRemodAdd': [year_remod_add],
        'MasVnrArea': [mas_vnr_area],
        'Fireplaces': [fireplaces],
        'BsmtFinSF1': [bsmt_fin_sf1],
        'LotArea': [lot_area],
        'WoodDeckSF': [wood_deck_sf],
        'OpenPorchSF': [open_porch_sf],
        'Neighborhood': [neighborhood],
        'HouseStyle': [house_style],
        'SaleCondition': [sale_condition]
    })

    prediction = model.predict(input_data)
    st.success(f'Предсказанная цена дома: ${prediction[0]:,.2f}')
    st.info(f'Примерный диапазон цены: ${prediction[0]*0.9:,.2f} - ${prediction[0]*1.1:,.2f}')