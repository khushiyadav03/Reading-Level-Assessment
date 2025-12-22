# Reading Level Assessment Tool

This project is a web-based application that assesses the reading level of a given text. It uses Natural Language Processing (NLP) techniques and a machine learning model to analyze text features and predict a readability score and corresponding grade level.

## Project Structure

- `app.py`: The main Flask application entry point.
- `train_model.py`: Script to train the machine learning model using the provided dataset.
- `download_nltk.py`: Helper script to download necessary NLTK data (stopwords, wordnet).
- `src/`: Directory containing source code modules.
    - `preprocessing.py`: Functions for text cleaning and lemmatization.
    - `features.py`: Functions to extract readability features (e.g., Flesch-Kincaid, SMOG).
    - `model.py`: Model training logic.
    - `utils.py`: Utility functions for model loading and score conversion.
- `templates/`: HTML templates for the web interface.
- `static/`: Static assets like CSS files.
- `requirements.txt`: List of Python dependencies.

## Setup and Installation

1.  **Install Dependencies**:
    Run the following command to install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download NLTK Data**:
    The project requires specific NLTK datasets. Run this script to download them:
    ```bash
    python download_nltk.py
    ```

3.  **Train the Model**:
    Before running the application, you must generate the trained model file (`model.pkl`). Run:
    ```bash
    python train_model.py
    ```
    This script reads the data, trains the regression model, and saves it to the root directory.

## Running the Application

1.  Start the Flask server:
    ```bash
    python app.py
    ```

2.  Open your web browser and navigate to:
    `http://127.0.0.1:5000`

## Usage

1.  Enter or paste text into the input area on the web page.
2.  Click the "Analyze Text" button.
3.  The application will display the calculated Readability Score and the predicted Grade Level.

## Methodology

The tool processes the input text through several stages:

1.  **Preprocessing**: The text is cleaned, and stop words are removed. Words are reduced to their base form (lemmatization).
2.  **Feature Extraction**: Various readability metrics are calculated, including Flesch-Kincaid Grade, SMOG Index, Automated Readability Index, and Coleman-Liau Index. Basic statistics like word count and average sentence length are also computed.
3.  **Prediction**: These features are fed into a trained regression model to predict the reading difficulty score.
