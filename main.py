"""
Functional Status Information analysis with unsupervised clustering methods for unlabeled data.
Author: Tara Jain
Student ID: acp24tkj
Date: 19/05/2025

Description:
This script performs an unsupervised clustering analysis with LDA and Kmeans on the MIMIC IV dataset with Pyspark intended to ensure scalability.

Requirements:
- PySpark
- Matplotlib
- Spacy
"""
import logging
import os

import spacy
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer
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
    tokens = [
        token.lemma_.lower() for token in doc
        if token.is_alpha and token.text.lower() not in stopwords and len(token.lemma_) > 2
    ]
    return tokens if tokens else None

def preprocess_sentences(unprocessed_sentences: DataFrame) -> DataFrame:
    """
    Preprocess sentences using spaCy:
    - Lowercase
    - Remove punctuation/non-alpha characters
    - Remove stopwords
    - Apply lemmatization
    - Drop rows with empty token lists
    Returns a DataFrame with 'filtered_sentences' and 'tokens'.
    """
    logging.info("Preprocessing sentences with spaCy NLP...")

    processed_sentences = unprocessed_sentences.withColumn(
        "tokens",
        tokenize_filter_lemmatize(col("sentence"))
    ).filter(
        col("tokens").isNotNull()
    ).cache()

    logging.info(f"Number of processed sentences: {processed_sentences.count()}")

    return processed_sentences

def _get_cbert_embeddings():
    pass

def extract_features(sentences, method = "tf", vocab_size: int = 5000, min_df: int = 2):
    # TODO(Done): Get features by using bag of words and frequency counts (LDA).
    if method == "tf":
        # Use CountVectorizer to create a bag of words representation

        vectorizer = CountVectorizer(inputCol="tokens", outputCol="features", vocabSize=vocab_size, minDF=min_df)

        model = vectorizer.fit(sentences)

        vectorized_data = model.transform(sentences)
        vectorized_data.cache()
        return vectorized_data, model.vocabulary

    # TODO(PO.2): Get features by using Clinical Bert embeddings
    elif method == "clinical_bert":
        # Use Clinical BERT embeddings
        pass
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tf' or 'clinical_bert'.")

def lda(sentences: DataFrame, vocabulary, k: int = 5, max_iter: int = 20):
    """
    Perform LDA on a DataFrame that contains a 'features' column.
    """
    logging.info(f"Running LDA with k={k}, maxIter={max_iter}")

    lda_model = LDA(k=k, maxIter=max_iter, seed=SEED, featuresCol="features").fit(sentences)

    perplexity = lda_model.logPerplexity(sentences)

    topics = lda_model.describeTopics()
    vocab = vocabulary
    topic_words = topics.rdd.map(lambda row: [vocab[i] for i in row['termIndices']]).collect()

    for idx, words in enumerate(topic_words):
        logging.info(f"Topic {idx}: {', '.join(words)}")

    return lda_model, perplexity

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

        # Extract term frequency features
    sentences_with_features, lda_vocab = extract_features(sentences, vocab_size=5000, min_df=2)

    lda_models = []
    perplexities = []

    # Try multiple values of k for LDA and compare perplexity
    for k in [3, 4, 5, 8]:
        model, perplexity = lda(sentences_with_features, lda_vocab, k=k)
        lda_models.append(model)
        perplexities.append(perplexity)
        logging.info(f"LDA Model with k={k} Perplexity: {perplexity:.4f}")


    # Convert to pandas dataframd and save to CSV for local testing
    # and debugging
    sentences_pd = sentences.toPandas()
    sentences_pd.to_csv("Outputs/sentences_preprocessed.csv", index=False)
    logging.info("Saved processed sentences to Outputs/sentences_preprocessed.csv")

    # TODO(P1): Perform NER to remove medication names or use a list. 
    
    # Extract term frequency features
    sentences_with_features = extract_features(sentences, vocab_size=5000, min_df=2)

    # TODO: Put this in a for loop with different hyperparameters


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
    