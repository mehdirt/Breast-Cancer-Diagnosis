import numpy as np
import pickle
import streamlit as st
import plotly.graph_objects as go
from utils.data_utils import get_data, clean_data

def add_sidebar() -> dict:
    """
    Create the Streamlit sidebar and collect user input via sliders.

    Returns:
        dict: A dictionary of user-provided input values from the sidebar.
    """
    st.sidebar.header("Cell Nuclei Measurements")

    data = clean_data(get_data(path="data/data.csv"))

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_data = {}

    for label, key in slider_labels:
        input_data[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )
    
    return input_data

def add_prediction(input_data: dict) -> None:
    """
    Load the trained model and scaler, predict the diagnosis, and display results.

    Args:
        input_data (dict): Dictionary of user-provided measurements.
    """
    model = pickle.load(open("model/model.pkl", "rb"))
    scalar = pickle.load(open("model/scalar.pkl", "rb"))

    array = np.array(list(input_data.values())).reshape(1, -1)
    scaled_array = scalar.transform(array)

    prediction = model.predict(scaled_array)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else: 
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write(f"Probability of being benign: {model.predict_proba(scaled_array)[0][0]:.2f}")
    st.write(f"Probability of being malignant: {model.predict_proba(scaled_array)[0][1]:.2f}")

    st.write("This app can assist medical professionals in making a diagnosis, but should not "
             "be used as a substitute for a professional diagnosis.")

def get_scaled_values(input_dict: dict, data_path: str) -> dict:
    """
    Scale the user input values based on the dataset's min-max range.

    Args:
        input_dict (dict): User-provided measurements.
        data_path (str): Path to the dataset.

    Returns:
        dict: Dictionary of scaled values.
    """
    data = clean_data(get_data(data_path))

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data: dict) -> go.Figure:
    """
    Generate a radar chart for the input data.

    Args:
        input_data (dict): User-provided measurements.

    Returns:
        go.Figure: A Plotly radar chart figure.
    """
    input_data = get_scaled_values(input_data, data_path="data/data.csv")

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean'],
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se'],
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst'],
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    return fig

def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="👨‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write(
            "This app predicts whether a breast mass is benign or malignant "
            "based on cytology lab measurements. Use the sliders in the sidebar to update measurements."
        )
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_prediction(input_data)

if __name__ == '__main__':
    main()
