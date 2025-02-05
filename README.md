# Breast Cancer Diagnosis

![image](https://github.com/user-attachments/assets/9e8e25b8-20b1-415b-ab6c-c42cfd633fec)


Welcome to the **Breast Cancer Diagnosis** project! This application uses machine learning to predict whether a breast tumor is malignant or benign based on a dataset of patient measurements. By combining data analysis, machine learning, and an intuitive web interface, this project demonstrates the power of AI in healthcare.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Details](#model-details)
4. [Web Application](#web-application)
5. [Technologies Used](#technologies-used)
6. [Getting Started](#getting-started)
7. [Project Demo](#project-demo)

---

## Project Overview

Breast cancer is one of the most common cancers among women worldwide. Early diagnosis and treatment can save lives. This project provides a tool to classify breast tumors as either **malignant** (cancerous) or **benign** (non-cancerous) using logistic regression. The model is deployed through a user-friendly web app built with Streamlit, allowing real-time predictions based on user inputs.

---

## Dataset

The model is trained on a dataset containing measurements of breast cancer cases. Each record includes various features derived from images of fine-needle aspirates of breast masses, such as:

- **Radius Mean**
- **Texture Mean**
- **Perimeter Mean**
- **Area Mean**
- **Smoothness Mean**
- **Compactness Mean**

These features were used to train the logistic regression model to predict the tumor type. The dataset is preprocessed, ensuring clean and normalized data for effective training.

---

## Model Details

- **Algorithm Used**: Logistic Regression
- **Framework**: Scikit-learn
- **Model Persistence**: The trained model is serialized using `pickle` for efficient reuse.

The logistic regression model was chosen for its simplicity, interpretability, and effectiveness in binary classification tasks. Model performance was validated using standard evaluation metrics.

---

## Web Application

The project features a web application where users can:

1. Input new data via interactive sliders for the specified features.
2. Get real-time predictions on whether the tumor is malignant or benign.
3. Visualize the results interactively using Plotly radar charts.

The application is hosted online for ease of access:
**[Visit the Web App](https://breast-cancer-diagnosis-rt.streamlit.app/)**

---

## Technologies Used

The project leverages the following technologies:

- **Python**: The core programming language for development.
- **Scikit-learn**: For building and training the logistic regression model.
- **Pickle**: For saving and loading the trained model.
- **Streamlit**: For creating the web interface.
- **Plotly**: For generating interactive data visualizations.

---

## Getting Started

To run the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/breast-cancer-diagnosis.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:

   ```bash
   streamlit run app/main.py
   ```

4. Open the application in your browser at `http://localhost:8501`.

---

## Project Demo

- Explore the web app to see the project in action: **[Live Demo](https://breast-cancer-diagnosis-rt.streamlit.app)**
- Users can adjust sliders corresponding to the tumor features and instantly view predictions.

  ![image](https://github.com/user-attachments/assets/6c836823-f7b3-46d4-8609-89ac0c78ce9d)

---
## Contact Information
Thank you for exploring the **Breast Cancer Diagnosis** project! If you have any feedback, feel free to reach out.

[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mahdirafati680@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahdi-rafati-97420a197/)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@mehdirt)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/mahdirafati)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/itsmehdirt)
