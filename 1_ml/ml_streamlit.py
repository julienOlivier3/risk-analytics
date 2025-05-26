import pickle

import pandas as pd
import streamlit as st


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
    st.title("Machine Learning Residual Value Prediction App 🚙")
    st.write("Upload a CSV file, map your column names, and get predictions!")

    # File uploader for CSV
    st.write("### 1. Upload your CSV file")
    uploaded_file = st.file_uploader(label="", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df_input = pd.read_csv(uploaded_file)
        st.markdown("Uploaded Data Preview:")
        st.dataframe(df_input.head())

        # Display input fields for column mapping
        st.write("### 2. Map your column names to the expected feature names")
        st.write("Provide the names of the columns in your CSV file that correspond to the expected feature names.")

        # Example expected feature names (replace with your actual feature names)
        expected_features = {
            'manufacturer': """The company that produces the vehicle, which can influence brand value and perceived
                reliability. **Example:** Toyota is often associated with reliability,
                which may enhance resale value.""",
            'model': """The specific name or designation of the vehicle, which can affect
                its market demand and resale value. **Example:** The Honda Civic is
                a popular model known for its fuel efficiency and reliability.""",
            'type': """The category of the vehicle, such as sedan, SUV, truck,
                etc., which impacts its utility and desirability. **Example:** An
                SUV like the Ford Explorer may have a higher residual value due
                to its popularity in family transport.""",
            'odometer': """The total distance the vehicle has traveled, measured in
                miles or kilometers, which is a key factor in determining wear and tear.
                **Example:** A vehicle with 30,000 miles is generally perceived to have
                less wear compared to one with 100,000 miles.""",
            'year': """The model year of the vehicle, indicating its age and often
                correlating with technological advancements and features.
                **Example:** A 2020 model may have advanced safety
                features not available in a 2015 model.""",
            'fuel': """The type of fuel the vehicle uses, such as gasoline,
                diesel, or electric, which can affect operating costs and environmental
                impact. **Example:** An electric vehicle like the Tesla Model 3 may
                retain value better due to growing interest in sustainability.""",
            'transmission': """The type of transmission system (automatic or manual)
                that can influence driving experience and maintenance costs. **Example:**
                Automatic transmissions are often preferred in urban settings,
                potentially increasing the vehicle's resale appeal.""",
            'cylinders': """The number of cylinders in the engine, which can affect
                performance, fuel efficiency, and insurance costs. **Example:** A V6 engine
                in a Chevrolet Silverado may offer a balance of power and fuel efficiency
                that appeals to truck buyers.""",
            'drive': """The drivetrain configuration (e.g., front-wheel drive,
                rear-wheel drive, all-wheel drive) that impacts handling and traction.
                **Example:** All-wheel drive vehicles are often more desirable in
                regions with harsh winters, like the Subaru Outback.""",
            'paint_color': """The color of the vehicle's exterior, which can influence
                buyer preferences and resale value due to trends in popularity. **Example:**
                Neutral colors like white or gray are generally more popular and may sell
                for a higher price compared to less common colors like bright green."""}
        column_mapping = {}

        for feature, description in expected_features.items():
            column_mapping[feature] = st.text_input(
                f"Column name for {feature.capitalize()}:",
                value=feature,
                key=feature,
                help=description,
                label_visibility="visible")

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
                df_output = df_input[expected_features.keys()].copy()
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
