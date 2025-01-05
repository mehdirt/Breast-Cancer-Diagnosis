import pandas as pd
import streamlit as st
import pickle

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="👨‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue \
                  sample. This app predicts using a machine learning model whether a breast mass is benign or \
                  malignant based on the measurements it receives from your cytosis lab. You can also update the \
                  measurements by hand using the sliders in the sidebar.")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        pass
    with col2:
        pass


if __name__ == '__main__':
    main()