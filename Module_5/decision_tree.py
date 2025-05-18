from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

df = pd.read_csv('weatherAUS.csv')

df = df.dropna(subset=['RainTomorrow'])

y = df['RainTomorrow']
features = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 
            'WindSpeed9am', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'RainToday']
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Location', 'RainToday']
numeric_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 
                    'WindSpeed9am', 'Humidity9am', 'Humidity3pm', 'Pressure9am']

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', categorical_pipeline, categorical_features),
    ('num', numeric_pipeline, numeric_features)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)

# joblib.dump(model_pipeline, 'random_forest_model.pkl')
joblib.dump(model_pipeline, 'random_forest_model.pkl', compress=('zlib', 3))
