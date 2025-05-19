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
from pyspark import SparkSession
import logging

DATA_PATH = "data/labeled_starter.csv"
SEED = 314159
OUTPUTS_PATH = "Outputs/"
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("Outputs", exist_ok=True)

def init_spark():
    return SparkSession.builder \
        .master("local[*]") \
        .appName("Fsi-nlp") \
        .getOrCreate()

def preprocessing():
    pass

def lda():
    pass

def kmeans():
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
    preprocessing()
    
    # TODO: Put this in a for loop with different hyperparameters
    lda()
    
    # TODO: Put this in a for loop with different hyperparameters
    kmeans()

    dimensionality_reduction()
    dimensionality_reduction()
    visualize_clusters_and_save()