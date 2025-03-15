import streamlit as st
import pickle
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from util import vin_to_year, vin_to_manufacturer, VINReplacer, ConditionalImputer, AgeCalculator, SportColumn, ColumnDropper, DataFrameSimpleImputer


@st.cache_resource
# Load the trained model
def load_model():
    model = pickle.load(open('models/ml_model.pkl', 'rb'))
    return model


@st.cache_resource
# Define the preprocessing and feature engineering pipeline
def load_preprocessing_pipeline():
    preprocessing_pipeline = pickle.load(open('pipelines/ml_model_pipeline.pkl', 'rb'))
    return preprocessing_pipeline


# Main Streamlit app
def main():
    st.title("Machine Learning Residual Value Prediction App")
    st.write("Upload a CSV file, map your column names, and get predictions!")

    # File uploader for CSV
    st.write("### 1. Upload your CSV file")
    uploaded_file = st.file_uploader(label="", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df_input = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df_input.head())

        # Display input fields for column mapping
        st.write("### 2. Map your column names to the expected feature names")
        st.write("Provide the names of the columns in your CSV file that correspond to the expected feature names.")
        
        # Example expected feature names (replace with your actual feature names)
        expected_features = ['manufacturer', 'model', 'type', 'odometer', 'year', 'fuel', 'transmission', 'cylinders', 'drive', 'paint_color']
        column_mapping = {}

        for feature in expected_features:
            column_mapping[feature] = st.text_input(f"Column name for '{feature}'", value=feature, key=feature)

        # Predict button
        if st.button("Predict"):
            # Validate column mappings
            if all(column_mapping.values()):
                # Rename columns in the uploaded data based on user input
                df_inference = df_input.rename(columns=column_mapping)

                # Add date of offering the car on used car market as today
                df_inference['posting_date'] = pd.to_datetime('today')

                # Load model and preprocessing pipeline
                preprocessing_pipeline = load_preprocessing_pipeline()
                model = load_model()

                # Preprocess the data
                try:
                    df_inference = preprocessing_pipeline.transform(df_inference)
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")

                # Perform predictions
                try:
                    predictions = model.predict(df_inference)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

                # Combine predictions with original data
                df_output = df_input[expected_features].copy()
                df_output['age'] = df_inference['age']
                df_output.insert(0, 'Prediction', predictions)

                # Display results
                st.write("### 3. Prediction samples")
                st.dataframe(df_output.head(5))
                st.write("Download the file below to get residual value predictions for all your samples.")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=df_output.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("Please provide all column mappings before proceeding.")


if __name__ == "__main__":
    main()