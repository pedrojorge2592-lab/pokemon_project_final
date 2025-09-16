import os, mlflow, streamlit as st, pandas as pd
st.set_page_config(page_title="Pokémon Legendary Predictor")
MODEL_URI = os.getenv("MODEL_URI")
if MODEL_URI:
    model = mlflow.pyfunc.load_model(MODEL_URI)
else:
    st.warning("Set MODEL_URI to your MLflow model path (see MLflow UI).")

st.title("Is this Pokémon Legendary?")
col1, col2, col3 = st.columns(3)
HP = col1.number_input("HP", 1, 255, 60)
Attack = col1.number_input("Attack", 1, 255, 70)
Defense = col1.number_input("Defense", 1, 255, 65)
SpA = col2.number_input("Sp. Atk", 1, 255, 70)
SpD = col2.number_input("Sp. Def", 1, 255, 70)
Speed = col2.number_input("Speed", 1, 255, 70)
Total = col3.number_input("Total", 6, 1125, HP+Attack+Defense+SpA+SpD+Speed)
Type1 = col3.text_input("Type 1", "Water")
Type2 = col3.text_input("Type 2 (optional)", "") or None

if st.button("Predict") and MODEL_URI:
    X = pd.DataFrame([{
        "HP": HP, "Attack": Attack, "Defense": Defense,
        "Sp. Atk": SpA, "Sp. Def": SpD, "Speed": Speed,
        "Total": Total, "Type 1": Type1, "Type 2": Type2
    }])
    y = model.predict(X)
    try:
        prob = float(y[0][1])
    except Exception:
        prob = float(y[0]) if hasattr(y, "__len__") else float(y)
    st.metric("Legendary probability", f"{prob:.2%}")
    st.write("Prediction:", "⭐ Legendary" if prob >= 0.5 else "Not Legendary")
