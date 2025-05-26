import torch
import pandas as pd
import streamlit as st
from bertopic import BERTopic

torch.classes.__path__ = []  # add this line to manually set it to empty.


@st.cache_resource
# Load the trained model
def load_model():
    model = BERTopic.load("models/topic_model", embedding_model="all-MiniLM-L6-v2")
    return model


# Main Streamlit app
def main():
    st.title("Insurance Claims Clustering App 💰")
    st.write("Upload a CSV file with claim descriptions and get associated insurance clusters!")

    # File uploader for CSV
    st.write("### 1. Upload your CSV file")
    uploaded_file = st.file_uploader(
        label="File containing claims descriptions to cluster",
        type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df_input = pd.read_csv(uploaded_file)
        st.markdown("Uploaded Data Preview:")
        st.dataframe(df_input.head())

        # Tick which columns to use for topic modeling
        st.write("### 2. Select columns for topic modeling")
        options = df_input.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to use for topic modeling", options)

    # Predict button
    if st.button("Create Clusters"):

        # Load topic model
        topic_model = load_model()

        df_input['Claim'] = df_input[selected_columns].agg(' '.join, axis=1)

        # Perform clustering
        try:
            topics, probs = topic_model.transform(df_input['Claim'].values)
        except Exception as e:
            st.error(f"Error during topic modeling: {e}")

        # Combine clusters with original data
        df_input['Topic'] = topics
        df_input['Probability'] = probs
        df_output = df_input.merge(
            topic_model.get_topic_info(),
            on='Topic',
            how='left')[["Claim", "Topic", "Representation", "Probability"]]

        # Display results
        st.write("### 3. Cluster samples")
        st.dataframe(df_output.head(5))
        st.write("Download the file below to get insurance clusters for all your claims.")
        st.download_button(
            label="Download Clusters as CSV",
            data=df_output.to_csv(index=False),
            file_name="clusters.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
