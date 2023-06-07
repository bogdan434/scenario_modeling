import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Flatten
import datetime
import os
import yfinance as yf
import plotly.graph_objects as go

import dash
from dash import dcc, dash_table
from dash import html
from dash.dependencies import Input, Output
from app_functions import *

'''Вказати шляхи до папок'''
# Шлях до файлу моделі
model_path = 'C:/Users/stopd/PycharmProjects/scenario_modeling/Model/model.h5'
# Шлях до файлу з даними
btc_data_path = 'C:/Users/stopd/PycharmProjects/scenario_modeling/Data/BTC.csv'
oil_data_path = 'C:/Users/stopd/PycharmProjects/scenario_modeling/Data/Brent Oil Futures Historical Data.csv'


# Функція для завантаження даних з API або з файлу
def load_data():
    if os.path.exists(btc_data_path) and os.path.exists(oil_data_path):
        # Завантаження даних з файлів
        bitcoin_data = pd.read_csv(btc_data_path)
        oil_data = pd.read_csv(oil_data_path)
    else:
        # Отримання даних з API
        btc = yf.download('BTC-USD', start='2010-01-01')
        oil = yf.download('BZ=F', start='2010-01-01')

        # Збереження даних у файли
        btc.to_csv(btc_data_path)
        oil.to_csv(oil_data_path)

        # Завантаження даних з файлів
        bitcoin_data = pd.read_csv(btc_data_path)
        oil_data = pd.read_csv(oil_data_path)

    # Перетворення колонки з датою в формат datetime
    bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
    oil_data['Date'] = pd.to_datetime(oil_data['Date'])

    # Об'єднання даних
    merged_data = pd.merge(bitcoin_data, oil_data, on='Date')

    return merged_data


# Функція для оновлення даних з API
def update_data():
    # Отримуємо дані про ціну BTC
    btc = yf.download('BTC-USD', start='2010-01-01')

    # Зберігаємо дані про ціну BTC у файл "BTC.csv"
    btc.to_csv(btc_data_path)

    # Отримуємо дані про ціну нафти (наприклад, Brent Crude Oil)
    oil = yf.download('BZ=F', start='2010-01-01')

    # Зберігаємо дані про ціну нафти у файл "Brent Oil Futures Historical Data.csv"
    oil.to_csv(oil_data_path)

    print("Дані оновлено успішно.")


# Функція для нормалізації даних
def normalize_data(data):
    # Нормалізація даних
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler


# Функція для створення матриці признаків
def create_features(bitcoin_data, oil_data, window_size):
    X, y = [], []
    for i in range(len(bitcoin_data) - window_size):
        X.append(np.concatenate((bitcoin_data[i:i + window_size], oil_data[i:i + window_size]), axis=1))
        y.append(bitcoin_data[i + window_size])
    return np.array(X), np.array(y)



# Функція для навчання моделі
def train_model(X_train, y_train, window_size):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(window_size, 2)))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=20, batch_size=16)

    return model


# Функція для зроблення прогнозу
def make_predictions(model, X_test, scaler, window_size, days):
    predictions = []
    current_features = X_test[0]

    for _ in range(len(X_test)):
        prediction = model.predict(current_features.reshape(1, window_size, 2), verbose=2)
        predictions.append(prediction[0][0])
        current_features = np.roll(current_features, -1, axis=0)
        current_features[-1] = prediction

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    future_predictions = []
    future_features = np.copy(X_test[-1])

    for _ in range(days):
        future_prediction = model.predict(future_features.reshape(1, window_size, 2))
        future_predictions.append(future_prediction[0][0])
        future_features = np.roll(future_features, -1, axis=0)
        future_features[-1] = future_prediction

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    return predictions.flatten(), future_predictions.flatten()


# Функція для виводу графіка прогнозу
def plot_predictions(test_prices, future_predictions, test_dates, future_dates):
    prediction_graph = go.Figure()
    prediction_graph.add_trace(go.Line(x=test_dates, y=test_prices, name="test data"))
    prediction_graph.add_trace(go.Line(x=future_dates, y=future_predictions, name="future predictions", mode="lines"))
    prediction_graph.update_layout(title_text='<b>Prediction<b>', title_x=0.5)
    return prediction_graph

def plot_btc_prices():
    data = load_data()
    bitc_graph = go.Figure()
    bitc_graph.add_trace(go.Line(x=data['Date'].values, y=data['Close_x'].values))
    bitc_graph.update_layout(title_text='<b>Bitcoin prices<b>', title_x=0.5)
    return bitc_graph

def plot_oil_prices():
    data = load_data()
    oil_graph = go.Figure()
    oil_graph.add_trace(go.Line(x=data['Date'].values, y=data['Close_y'].values))
    oil_graph.update_layout(title_text='<b>Oil prices<b>', title_x=0.5)
    return oil_graph


# Функція для обчислення середньої абсолютної помилки
def calculate_mae(predictions, y_test):
    mae = np.mean(np.abs(predictions - y_test))
    print(f'Середня абсолютна помилка (MAE): {mae}')


# Функція для збереження моделі
def save_model(model):
    model.save(model_path)
    print('Модель збережена успішно.')


# Функція для завантаження моделі
def load_saved_model():
    try:
        model = load_model('C:/Users/stopd/PycharmProjects/scenario_modeling/Model/model.h5')
        print('Модель завантажена успішно.')
        return model
    except:
        print('Модель не знайдена.')
        return None


# Функція для виводу інформації про останнє оновлення моделі
def show_last_updated():
    try:
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
        print('Останнє оновлення моделі:', mod_time)
    except:
        print('Модель не знайдена.')


# Функція для побудови графіка кореляції цін за роками
def plot_price_correlation():
    data = load_data()
    data['Year'] = data['Date'].dt.year
    years = sorted(data['Year'].unique())
    correlations = []

    for year in years:
        year_data = data[data['Year'] == year]
        correlation = year_data['Close_x'].corr(year_data['Close_y'])
        correlations.append(correlation)

    prices_graph = go.Figure()
    prices_graph.add_trace(go.Line(x=years, y=correlations, marker_color='rgb(171, 226, 251)')

                           )
    prices_graph.update_layout(title_text='<b>Prices correlation by year<b>', title_x=0.5)
    return prices_graph


# Функція для побудови графіка кореляції цін помісячно
def plot_monthly_price_correlation(data):
    data['Month'] = data['Date'].dt.month
    months = sorted(data['Month'].unique())
    correlations = []

    for month in months:
        month_data = data[data['Month'] == month]
        correlation = month_data['Close_x'].corr(month_data['Close_y'])
        correlations.append(correlation)

    plt.plot(months, correlations)
    plt.xlabel('Місяць')
    plt.ylabel('Кореляція')
    plt.title('Зміна кореляції цін помісячно')
    plt.show()