import joblib
aussie_rain = joblib.load('random_forest_model.pkl')
joblib.dump(aussie_rain, 'random_forest_model.pkl', compress=('zlib', 3))