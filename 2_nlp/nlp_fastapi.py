from contextlib import asynccontextmanager
import pandas as pd
import fastapi
from pydantic import BaseModel
from bertopic import BERTopic


# Function to load the BERTopic model
def load_model():
    model = BERTopic.load("models/topic_model", embedding_model="all-MiniLM-L6-v2")
    return model


# Dictionary to store the loaded model and topic information
model = {}


# Lifespan context manager to load and clean up the model
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Load the BERTopic model and topic information
    model['model'] = load_model()
    model['topic_info'] = model['model'].get_topic_info()
    yield
    # Clean up the ML models and release the resources
    model.clear()


# Pydantic model for input data
class Claim(BaseModel):
    claim_description: str


# Create a FastAPI app with the lifespan context
app = fastapi.FastAPI(lifespan=lifespan)


# Health check endpoint
@app.get("/health")
async def check_health():
    return {"Server": "I am healthy!"}


# Endpoint for clustering
@app.post("/v1/cluster")
async def create_cluster(input_data: Claim):
    # Preprocess the data
    input_dict = input_data.model_dump()  # Corrected: No 'mode' argument
    df_inference = pd.DataFrame([input_dict])

    # Perform clustering
    try:
        print("Performing clustering...")
        topics, probs = model['model'].transform(df_inference.claim_description.values)
        print("Topics:", topics)

        # Retrieve the topic representation
        topic = model['topic_info'].loc[
            model['topic_info']['Topic'] == topics[0], 'Representation'
        ].values[0]  # Use `.values[0]` to extract the string representation
        print("Topic Representation:", topic)
        print("Probabilities:", probs)

        return {
            "topic": topic,
            "probability": probs.tolist()  # Convert numpy array to list for JSON serialization
        }
    except Exception as e:
        return {"error": f"Error during prediction: {e}"}
