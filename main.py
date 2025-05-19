"""
Functional Status Information An unsupervised clustering analysis for unlabeled data
Author: Tara Jain
Student ID: acp24tkj
Date: 19/05/2025

Description:
This script performs an unsupervised clustering analysis on the MIMIC IV dataset with Pyspark intended to ensure scalability.

Requirements:
- PySpark
- Matplotlib
"""
import logging
import os

import spacy
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType

DATA_PATH = "data/labeled_starter.csv"
SEED = 314159
OUTPUTS_PATH = "Outputs/"
# Configure logger
os.makedirs(OUTPUTS_PATH, exist_ok=True)
log_file_path = os.path.join(OUTPUTS_PATH, "logs.txt")
if not os.path.exists(log_file_path):
    with open(log_file_path, "w") as f:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
os.makedirs("Outputs", exist_ok=True)

def init_spark():
    return SparkSession.builder \
        .master("local[*]") \
        .appName("Fsi-nlp") \
        .getOrCreate()

# Load spaCy once globally
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Basic English stopwords from spaCy
stopwords = nlp.Defaults.stop_words

@udf(ArrayType(StringType()))
def tokenize_filter_lemmatize(sentence: str):
    if not sentence:
        return []
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in stopwords]
    return tokens if tokens else None

def preprocess_sentences(unprocessed_sentences: DataFrame) -> DataFrame:
    """
    Preprocess sentences using spaCy:
    - Lowercase
    - Remove punctuation/non-alpha characters
    - Remove stopwords
    - Apply lemmatization
    - Drop rows with empty token lists
    Returns a DataFrame with 'filtered_sentences' and 'tokenized'.
    """
    logging.info("Preprocessing sentences with spaCy NLP...")

    processed_sentences = unprocessed_sentences.withColumn(
        "tokenized",
        tokenize_filter_lemmatize(col("sentence"))
    ).filter(
        col("tokenized").isNotNull()
    )

    logging.info(f"Number of processed sentences: {processed_sentences.count()}")

    return processed_sentences

def _get_cbert_embeddings():
    pass

def extract_features():
    # TODO(PO.1): Get features by using bag of words and frequency counts (LDA).
    # TODO(PO.2): Get features by using Clinical Bert embeddings
    pass

def lda():
    # TODO(P1): Implement LDA, might need to use non-semantic embeddings.
    pass

def kmeans():
    # TODO(P0): Implement K-means clustering
    pass

def evaluate_clusters():
    """Get cluster purity or eval."""
    pass

def dimensionality_reduction():
    pass

def visualize_clusters_and_save():
    """This function creates a 3D scatter plot of the clusters and saves the datapoints belonging to each cluster in a CSV file."""

def main():

    spark = init_spark()
    # sentences = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    
    # Read in only the first 2000 records for development and test.
    sentences = spark.read.csv(DATA_PATH, header=True, inferSchema=True).limit(200)

    logging.info(f"Number of sentences in dataframe: {sentences.count()}")

    sentences = preprocess_sentences(sentences)

    row_count = sentences.count()
    logging.info(f"[DEBUG] Row count before writing CSV: {row_count}")


    # Convert to pandas dataframd and save to CSV for local testing
    # and debugging
    # sentences_pd = sentences.toPandas()
    # sentences_pd.to_csv("Outputs/sentences_preprocessed.csv", index=False)
    # logging.info("Saved processed sentences to Outputs/sentences_preprocessed.csv")

    # TODO: Appends a column with features comprising of words extracted from the sentences to the Spark DataFrame - sentences. 
    extract_features()
    
    # TODO: Put this in a for loop with different hyperparameters
    lda()

    ## LDA ## 
    # On LDA best fit
    dimensionality_reduction()
    evaluate_clusters()
    visualize_clusters_and_save()

    ## K-means ##
    # TODO: Put this in a for loop with different hyperparameters
    kmeans()
    dimensionality_reduction()
    evaluate_clusters()

    visualize_clusters_and_save()
    spark.stop()

if __name__ == "__main__":
    main()
    