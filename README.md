# 🫀 Heart Disease Prediction Project

This project predicts the likelihood of heart disease using the **UCI Cleveland Heart Disease Dataset**.  
It covers the full machine learning pipeline: **data preprocessing, PCA, feature selection, supervised & unsupervised learning, hyperparameter tuning**, and a **Streamlit web app** for real-time predictions.


File Structure
 Heart_Disease_Project/
 ├── data/ # Dataset
 ├── notebooks/ # Step-by-step ML pipeline
 ├── models/ # Saved model (final_model.pkl)
 ├── ui/ # Streamlit app (app.py)
 ├── results/ # Plots & metrics
 ├── requirements.txt # Dependencies
 └── README.md

 Setup
```bash
git clone https://github.com/mosam22/Heart_Disease_Project.git
cd Heart_Disease_Project
python -m venv venv
venv\Scripts\activate  
pip install -r requirements.txt

Run the Project
1. Data Preprocessing & Training
 python notebooks\01_data_preprocessing.py
 python notebooks\02_pca_analysis.py
 python notebooks\03_feature_selection.py
 python notebooks\04_supervised_learning.py
 python notebooks\05_unsupervised_learning.py
 python notebooks\06_hyperparameter_tuning.py
 python notebooks\07_save_and_test_model.py
  a trained model will be saved at:
  models/final_model.pkl

2.Launch Streamlit Web App
  cd ui
  streamlit run app.py
 Then open http://localhost:8501
 in your browser.

Features
 Data Cleaning & Preprocessing
 PCA (Principal Component Analysis)
 Feature Selection (Random Forest + RFE)
 Supervised Learning (LogReg, Decision Tree, RF, SVM)
 Unsupervised Learning (KMeans, Hierarchical Clustering)
 Hyperparameter Tuning (GridSearchCV)
 Deployment with Streamlit

Dataset
 UCI Heart Disease Dataset (Cleveland subset)
 Source: UCI Machine Learning Repository

Future Improvements
 Deploy app on cloud (Heroku / Render)
 Add more advanced models (XGBoost, LightGBM)
 Cross-validation with larger datasets





