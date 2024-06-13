import joblib
import streamlit as st
import pandas as pd

# Carica il modello
model_pipe = joblib.load('penguinspipe.pkl')
st.image("pingo.jpeg", use_column_width=True)
st.title('Indovina il Pinguino')

# Input dall'utente
island = st.selectbox("Isola", ["Biscoe", "Dream", "Torgersen"])
bill_length_mm = st.number_input("Lunghezza del becco (mm)", min_value=0.0, step=0.1)
bill_depth_mm = st.number_input("Profondità del becco (mm)", min_value=0.0, step=0.1)
flipper_length_mm = st.number_input("Lunghezza delle pinne (mm)", min_value=0.0, step=0.1)
body_mass_g = st.number_input("Massa corporea (g)", min_value=0.0, step=1.0)
sex = st.selectbox("Sesso", ["Male", "Female"])

# Prepara i dati per la predizione
data = {
    "island": [island],
    "bill_length_mm": [bill_length_mm],
    "bill_depth_mm": [bill_depth_mm],
    "flipper_length_mm": [flipper_length_mm],
    "body_mass_g": [body_mass_g],
    "sex": [sex],
}

input_df = pd.DataFrame(data)

# Effettua la predizione
if st.button('**Indovina**'):
    res = model_pipe.predict(input_df).astype(int)[0]

    classes = {0: 'Adelie', 1: 'Gentoo', 2: 'Chinstrap'}
    y_pred = classes[res]

    st.write(f"Il pinguino è della specie: {y_pred}")
