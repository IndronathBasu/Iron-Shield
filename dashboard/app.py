import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

import sys
sys.path.append(".")

from pipeline.predict_pipeline import run_pipeline

HISTORY_FILE = "user_history.csv"


st.set_page_config(
    page_title="Iron Shield",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Iron Shield")
st.subheader("AI-Based Non-Invasive Anemia Risk Detection")

st.write(
    "Upload an eye image to estimate hemoglobin levels "
    "and detect anemia risk."
)

uploaded_file = st.file_uploader(
    "Upload Eye Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    os.makedirs("temp", exist_ok=True)

    img_path = f"temp/{uploaded_file.name}"

    with open(img_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Iron Shield AI"):

        with st.spinner("Analyzing image..."):

            hb, risk = run_pipeline(img_path)

        st.success("Analysis Complete")

        st.metric("Predicted Hemoglobin", f"{hb:.2f} g/dL")
        st.metric("Risk Level", risk)

        today = datetime.date.today()

        new_data = pd.DataFrame(
            [[today, hb, risk]],
            columns=["date","hb","risk"]
        )

        if os.path.exists(HISTORY_FILE):

            history = pd.read_csv(HISTORY_FILE)

            history = pd.concat([history,new_data])

        else:

            history = new_data

        history.to_csv(HISTORY_FILE,index=False)


# Hb Tracking Section
st.subheader("📈 Hemoglobin Progression")

if os.path.exists(HISTORY_FILE):

    history = pd.read_csv(HISTORY_FILE)

    history["date"] = pd.to_datetime(history["date"])

    fig, ax = plt.subplots()

    ax.plot(history["date"], history["hb"], marker="o")

    ax.set_ylabel("Hemoglobin (g/dL)")
    ax.set_xlabel("Date")
    ax.set_title("Hb Improvement Over Time")

    st.pyplot(fig)

else:

    st.info("No previous predictions yet.")