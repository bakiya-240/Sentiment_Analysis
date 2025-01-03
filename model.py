import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model
def load_model():
    try:
        model = joblib.load('sentiment_model.pkl')
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error("Model file 'sentiment_model.pkl' not found.")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

# Predict sentiment based on input text
def predict_sentiment(text):
    try:
        model = load_model()  # Load the model
        prediction = model.predict([text])  # Predict sentiment
        sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}  # Adjust as per your mapping
        sentiment = sentiment_map.get(prediction[0], 'Unknown')  # Return the sentiment label
        logging.info(f"Text: {text} | Sentiment: {sentiment}")
        return sentiment
    except Exception as e:
        logging.error(f"Error predicting sentiment: {e}")
        return "Error"

# Test the functionality
if __name__ == "__main__":
    sample_text = "The product is amazing and exceeded expectations!"
    print(f"Input Text: {sample_text}")
    print(f"Predicted Sentiment: {predict_sentiment(sample_text)}")
