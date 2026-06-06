import streamlit as st
import numpy as np
import joblib

# ── Load saved model artifacts ────────────────────────────────
model   = joblib.load('covid_model.pkl')
scaler  = joblib.load('covid_scaler.pkl')
encoder = joblib.load('continent_encoder.pkl')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title='COVID-19 Death Predictor', page_icon='🦠', layout='centered')

st.title('🦠 COVID-19 Death Predictor')
st.markdown('Powered by **Gradient Boosting** — R² : `0.9546` | MAE : `1,243` | RMSE : `2,497`')
st.divider()

# ── Input form ────────────────────────────────────────────────
st.subheader('Enter Country Data')

col1, col2 = st.columns(2)

with col1:
    population = st.number_input('Population',  min_value=0.0, value=1_000_000.0, step=1000.0, format='%f')
    cases      = st.number_input('Cases',       min_value=0.0, value=50000.0,     step=100.0)
    recovered  = st.number_input('Recovered',   min_value=0.0, value=45000.0,     step=100.0)

with col2:
    tests      = st.number_input('Tests',       min_value=0.0, value=200000.0,    step=1000.0)
    continent  = st.selectbox('Continent', [
        'Africa', 'Asia', 'Europe',
        'North-America', 'Oceania', 'South-America'
    ])

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button('Predict Deaths', use_container_width=True, type='primary'):

    # Derived features (must match training order)
    cases_per_million  = (cases  / population * 1_000_000) if population else 0
    tests_per_million  = (tests  / population * 1_000_000) if population else 0
    mortality_rate     = 0   # unknown at prediction time
    recovery_rate      = (recovered / cases) if cases else 0
    test_positivity    = (cases / tests)     if tests else 0
    continent_encoded  = encoder.transform([continent])[0]

    features = np.array([[
        population, cases, recovered, tests,
        cases_per_million, tests_per_million,
        mortality_rate, recovery_rate, test_positivity,
        continent_encoded,
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    # ── Display result ────────────────────────────────────────
    st.divider()
    st.subheader('Prediction Result')

    col_a, col_b, col_c = st.columns(3)
    col_a.metric('Predicted Deaths',  f'{prediction:,.0f}')
    col_b.metric('Mortality Rate',    f'{(prediction / cases * 100) if cases else 0:.2f}%')
    col_c.metric('Recovery Rate',     f'{(recovered / cases * 100) if cases else 0:.2f}%')

    if prediction < 1000:
        st.success('Low death count predicted.')
    elif prediction < 10000:
        st.warning('Moderate death count predicted.')
    else:
        st.error('High death count predicted.')
