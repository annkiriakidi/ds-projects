import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('🌧️ Прогноз дощу на завтра')

st.markdown('Введіть характеристики погоди для отримання прогнозу.')

Location = st.selectbox('Локація', ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'])
MinTemp = st.number_input('Мінімальна температура (°C)', -10.0, 50.0, 15.0)
MaxTemp = st.number_input('Максимальна температура (°C)', -10.0, 50.0, 25.0)
Rainfall = st.number_input('Кількість опадів (мм)', 0.0, 500.0, 0.0)
WindGustSpeed = st.number_input('Швидкість пориву вітру (км/год)', 0.0, 200.0, 35.0)
WindSpeed9am = st.number_input('Швидкість вітру о 9:00 (км/год)', 0.0, 100.0, 15.0)
Humidity9am = st.slider('Вологість о 9:00 (%)', 0, 100, 60)
Humidity3pm = st.slider('Вологість о 15:00 (%)', 0, 100, 50)
Pressure9am = st.number_input('Тиск о 9:00 (гПа)', 900.0, 1100.0, 1012.0)
RainToday = st.selectbox('Чи йшов дощ сьогодні?', ['Yes', 'No'])

model_pipeline = joblib.load('random_forest_model.pkl')

input_data = pd.DataFrame({
    'Location': [Location],
    'MinTemp': [MinTemp],
    'MaxTemp': [MaxTemp],
    'Rainfall': [Rainfall],
    'WindGustSpeed': [WindGustSpeed],
    'WindSpeed9am': [WindSpeed9am],
    'Humidity9am': [Humidity9am],
    'Humidity3pm': [Humidity3pm],
    'Pressure9am': [Pressure9am],
    'RainToday': [RainToday]
})

if st.button('Зробити прогноз'):
    prediction = model_pipeline.predict(input_data)[0]
    probabilities = model_pipeline.predict_proba(input_data)[0]

    label = 'Так' if prediction == 'Yes' else 'Ні'
    probability = probabilities[1] if prediction == 'Yes' else probabilities[0]

    st.subheader(f'🌧️ Прогноз: {label}')
    st.write(f'Ймовірність: {probability:.2%}')
