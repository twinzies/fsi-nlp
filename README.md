# FSI-NLP

An exploratory natural language processing pipeline for **Functional Status
Information (FSI)** in clinical text. The project analyzes sentences from
MIMIC-IV-derived data using unsupervised topic modeling and clustering, with
PySpark handling preprocessing and model training.

The pipeline compares two approaches:

- **LDA:** term-frequency vectors reveal topics, with each sentence assigned to
  its highest-probability topic.
- **KMeans:** ClinicalBERT embeddings group sentences by semantic similarity.

Results include silhouette scores, LDA log-perplexity, 3D t-SNE visualizations,
and LDA topic word clouds. Although the input CSVs contain labels, the current
pipeline does not use them for training or evaluation.

## Repository layout

```text
fsi-nlp/
|-- main.py                       # Preprocessing, features, models, and reporting
|-- script.sh                     # Launcher for an existing Conda environment
|-- data/
|   |-- labeled_starter.csv        # Default input: sentence and labels columns
|   |-- cleaned_first350.csv       # Alternative dataset; not loaded by default
|   `-- multi_category.csv         # Alternative dataset; sentence and label columns
|-- Outputs/
|   |-- clustering_evaluation.csv  # Model comparison metrics
|   |-- logs.txt                   # Runtime messages
|   |-- tsne_clusters_*.csv         # Reduced coordinates and cluster assignments
|   |-- tsne_clusters_*.png         # 3D scatter plots
|   |-- wordclouds_lda_k=*/         # Word clouds for each LDA topic
|   `-- ...                        # Saved preprocessing and clustering artifacts
|-- LICENSE                       # MIT license
`-- README.md
```

Some committed outputs are artifacts from earlier runs; not all are regenerated
by the current script.

## System Architecture

```text
CSV sentences
      |
      v
Local PySpark session (local[*])
      |
      v
spaCy tokenization, stopword filtering, and lemmatization
      |
      +--> CountVectorizer (term frequency) --> LDA (k = 4, 5, 6)
      |
      `--> Bio_ClinicalBERT embeddings ------> KMeans (k = 3, 4, 5)
                         |
                         v
          Evaluation and reporting --> Outputs/
```

This is a batch script, not a web service. Spark uses all available local CPU
cores. ClinicalBERT runs through PyTorch; evaluation, t-SNE, and plotting collect
data into the Python driver using pandas and scikit-learn, so these stages are
limited by local memory.

## Setup

Use Python 3.12 and a Java installation compatible with your PySpark version
(Java 17 is a suitable starting point). Ensure `java` is on your `PATH`, and set
`JAVA_HOME` if Spark cannot locate it.

From a local checkout, create and activate a virtual environment:

```bash
cd fsi-nlp
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyspark matplotlib numpy pandas spacy torch scikit-learn transformers wordcloud
python -m spacy download en_core_web_sm
```

The repository does not currently include a dependency manifest or pinned
versions. The commands above install the packages imported by `main.py`; they
are not a version-locked environment.

The first ClinicalBERT run requires internet access to download
`emilyalsentzer/Bio_ClinicalBERT` from Hugging Face. Subsequent runs can reuse the
local model cache. A GPU is not required; the current code uses CPU inference.

## Quickstart

Run from the repository root so the relative input and output paths resolve:

```bash
source .venv/bin/activate
export PYSPARK_PYTHON="$(command -v python)"
export MPLBACKEND=Agg
spark-submit main.py
```

`MPLBACKEND=Agg` allows plots to be saved without a graphical display.

By default, the script reads `data/labeled_starter.csv` and processes at most
**1,000 rows**. Input must have a header and a `sentence` column containing text.
To use another dataset or change the row limit, edit `DATA_PATH` and the
`.limit(1000)` call in `main.py`. Model choices, cluster counts, and the random
seed (`314159`) are also configured in the source; there are no command-line
options.

Inspect `Outputs/clustering_evaluation.csv` for comparisons and
`Outputs/tsne_clusters_*.png` for visualizations. Runtime messages are written
to the console and `Outputs/logs.txt`. Repeated runs overwrite matching output
files, including the log.

If you already have the Conda environment `machine-learning-312-env`, the
provided `bash script.sh` launcher activates it and runs Spark. Otherwise, use
the virtual-environment quickstart above.

## Metrics

The pipeline records the following unsupervised evaluation metrics in
`Outputs/clustering_evaluation.csv`:

- **Silhouette score:** Measures how well each sentence fits its assigned
  cluster compared with other clusters. Values range from -1 to 1; higher
  values generally indicate better-separated clusters. For LDA, this is
  calculated from topic-distribution vectors; for KMeans, it is calculated
  from ClinicalBERT embeddings.
- **LDA log-perplexity:** Measures how well the fitted LDA model explains the
  input term-frequency data. Lower values indicate better fit on that data.
  This value is recorded for LDA runs only.

The CSV contains an `algorithm` and `k` for each run, plus `silhouette` and,
for LDA, `perplexity`. These scores compare unsupervised model behavior; they
do not measure clinical accuracy or agreement with the dataset labels, which
the current pipeline does not use.

## Current limitations

- KMeans word clouds are currently skipped: its reporting call omits the
  `tokens` column required by the word-cloud helper. LDA word clouds are generated.
- t-SNE uses its default perplexity of 30, so more than 30 processed sentences
  are required. Silhouette evaluation also requires multiple populated clusters
  and fewer clusters than samples.
- Local collection and repeated ClinicalBERT inference can make larger runs
  memory-intensive and slow, despite the use of Spark.

Use clinical data only under its applicable access and usage terms. The
repository's MIT license does not replace dataset permissions. This project is
for exploratory research, not clinical decision-making.

## Author

**Tara Jain**, credited in `main.py`.

## License

Source code is available under the [MIT License](LICENSE).
