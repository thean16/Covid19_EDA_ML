import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib

df = pd.read_csv("covid_19.csv")
df = df.dropna(subset=['continent', 'population'])
df['Deaths'] = df['Deaths'].fillna(0)
df['Recovered'] = df['Recovered'].fillna(0)
df['Tests'] = df['Tests'].fillna(0)

def cap_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col + '_capped'] = df[col].clip(lower=lower, upper=upper)
    return df

for col in ['Cases', 'Deaths', 'Recovered', 'Tests']:
    df = cap_outliers_iqr(df, col)

df_ml = df.copy()
for col in ['Cases', 'Deaths', 'Recovered', 'Tests']:
    df_ml[col] = df_ml[col + '_capped']
    df_ml.drop(columns=[col + '_capped'], inplace=True)

pop = df_ml['population'].replace(0, np.nan)
df_ml['cases_per_million']  = (df_ml['Cases']  / pop) * 1_000_000
df_ml['tests_per_million']  = (df_ml['Tests']  / pop) * 1_000_000
df_ml['mortality_rate']     = df_ml['Deaths']    / df_ml['Cases'].replace(0, np.nan)
df_ml['recovery_rate']      = df_ml['Recovered'] / df_ml['Cases'].replace(0, np.nan)
df_ml['test_positivity']    = df_ml['Cases']     / df_ml['Tests'].replace(0, np.nan)

le = LabelEncoder()
df_ml['continent_encoded'] = le.fit_transform(df_ml['continent'].fillna('Unknown'))

features = [
    'population', 'Cases', 'Recovered', 'Tests',
    'cases_per_million', 'tests_per_million',
    'mortality_rate', 'recovery_rate', 'test_positivity',
    'continent_encoded'
]
target = 'Deaths'

df_ml = df_ml[features + [target]].fillna(0)
X = df_ml[features]
y = df_ml[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model,  'covid_model.pkl')
joblib.dump(scaler, 'covid_scaler.pkl')
joblib.dump(le,     'continent_encoder.pkl')

print(f"numpy version: {np.__version__}")
print("Models saved successfully.")
