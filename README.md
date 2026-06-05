# Cricket Match Winner Prediction and Semantic Search System

## Project Overview

This project was developed as part of a Python, Web Scraping, and Machine Learning assignment.

The project consists of three tasks:

1. Web Scraping Cricket Match Data
2. Machine Learning-Based Match Winner Prediction
3. Semantic Search using Vector Embeddings (Bonus Task)

The system collects cricket match information from publicly available cricket statistics websites, stores the data in CSV format, trains a machine learning model to predict match winners, and implements a semantic search system that retrieves relevant matches using natural language queries.

---

# Project Structure

```text
.
├── fallback_data.py
├── match_data.csv
├── model.py
├── models
│   ├── team_encoder.pkl
│   ├── venue_encoder.pkl
│   ├── winner_encoder.pkl
│   └── winner_predictor.pkl
├── prediction.py
├── rag_search.py
├── README.md
├── requirements.txt
└── scraper.py
```

---

# Task 1: Web Scraping Cricket Match Data

## Objective

The objective of this task is to automatically collect cricket match information from a publicly available cricket statistics website using Python.

## Data Collected

The scraper extracts the following details for the last 10 completed matches involving two selected teams:

* Match Date
* Team 1 Name
* Team 2 Name
* Venue
* Match Winner
* Top Scorer
* Top Scorer Runs

## Technologies Used

* Python
* Requests
* BeautifulSoup
* Pandas

## Output

The extracted data is saved in:

```text
match_data.csv
```

## Running the Scraper

Install required dependencies:

```bash
pip install pandas requests beautifulsoup4
```

Run the script:

```bash
python scraper.py
```

After execution, a CSV file containing the scraped match information will be generated.

---

# Task 2: Machine Learning Match Winner Prediction

## Objective

Build a machine learning model that predicts the winner of a cricket match using historical match data.

## Dataset

The model was trained using the dataset generated from Task 1.

Dataset Summary:

```text
Total Matches Loaded: 20
```

## Data Preprocessing

The following preprocessing steps were performed:

* Loaded data using Pandas
* Removed missing values
* Standardized team and venue information
* Encoded categorical variables using Label Encoding
* Created numerical features suitable for machine learning

## Features Used

| Feature   | Description              |
| --------- | ------------------------ |
| team1_enc | Encoded Team 1           |
| team2_enc | Encoded Team 2           |
| venue_enc | Encoded Venue            |
| is_home   | Home advantage indicator |

## Target Variable

The target variable is the match winner.

## Machine Learning Algorithm

### Random Forest Classifier

Random Forest was selected because:

* It performs well on structured tabular data.
* It handles non-linear relationships effectively.
* It provides feature importance scores.
* It generally performs better than a single decision tree.

## Model Evaluation

### Accuracy

```text
Accuracy: 0.50
```

The model correctly predicted 50% of the matches in the test dataset.

### Classification Report

```text
              precision    recall  f1-score   support

   Australia       1.00      0.33      0.50         3
       India       0.50      1.00      0.67         1
    Pakistan       0.00      0.00      0.00         0

    accuracy                           0.50         4
   macro avg       0.50      0.44      0.39         4
weighted avg       0.88      0.50      0.54         4
```

## Discussion

The dataset contains only 20 matches, which is relatively small for training a predictive model.

As more historical match data is collected, the model's performance is expected to improve significantly.

The current implementation demonstrates the complete machine learning workflow:

* Data preparation
* Feature engineering
* Model training
* Model evaluation
* Model persistence

## Feature Importance

The Random Forest model identified the following feature importance scores:

| Feature   | Importance Score |
| --------- | ---------------- |
| venue_enc | 0.4324           |
| team2_enc | 0.2773           |
| team1_enc | 0.2025           |
| is_home   | 0.0877           |

### Interpretation

* Venue was the most influential feature.
* Team combinations had a strong impact on match outcomes.
* Home advantage contributed the least in this dataset.

## Saved Model Files

After training, the following files are generated:

```text
winner_predictor.pkl
team_encoder.pkl
venue_encoder.pkl
winner_encoder.pkl
```

### Purpose

| File                 | Description                 |
| -------------------- | --------------------------- |
| winner_predictor.pkl | Trained Random Forest model |
| team_encoder.pkl     | Team label encoder          |
| venue_encoder.pkl    | Venue label encoder         |
| winner_encoder.pkl   | Winner label encoder        |

These files allow predictions to be made without retraining the model.

## Running the Model

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

Run:

```bash
python model.py
```

The script will:

1. Load the dataset
2. Preprocess data
3. Train the model
4. Evaluate performance
5. Display feature importance
6. Save the trained model

---

# Task 3: Semantic Search Using Vector Embeddings

## Objective

Implement a semantic search system that retrieves cricket matches based on the meaning of a user's query rather than exact keyword matching.

## What is Semantic Search?

Semantic search converts text into numerical vector representations called embeddings.

Instead of searching for exact words, the system compares the meaning of the query with stored match descriptions and returns the most relevant matches.

## Example Match Description

Each match record is converted into a sentence:

```text
India vs Australia at Melbourne on 15 Jan 2024.
Australia won.
Top scorer: David Warner with 87 runs.
```

## Technologies Used

* Sentence Transformers
* ChromaDB
* Pandas

## Embedding Model

```text
all-MiniLM-L6-v2
```

This model converts each match description into a dense vector representation.

## Workflow

### Step 1

Convert structured match data into natural language sentences.

### Step 2

Generate embeddings using Sentence Transformers.

### Step 3

Store embeddings in:

* In-memory vector storage, or
* ChromaDB vector database

### Step 4

Accept user queries and perform similarity search.

### Example Query

```text
Show me matches where the away team won
```

### Output

The system returns the top 3 most relevant matches based on semantic similarity.

## Running Semantic Search

Install dependencies:

```bash
pip install sentence-transformers chromadb pandas
```

Run:

```bash
python rag_search.py
```

Example:

```text
Enter Query:
Show me matches where Australia won away from home
```

The program returns the three most relevant match records.

---

# Dependencies

Install all required packages:

```bash
pip install pandas numpy requests beautifulsoup4 scikit-learn joblib sentence-transformers chromadb
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

---

# Conclusion

This project demonstrates practical skills in web scraping, data preprocessing, machine learning, model evaluation, vector embeddings, and semantic search. It follows a complete end-to-end workflow from data collection to intelligent information retrieval and serves as a foundation for more advanced sports analytics systems.
