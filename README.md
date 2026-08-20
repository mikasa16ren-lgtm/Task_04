# VeriFact: Fake News Statistical Analysis and Data Preparation

This project contains four related tasks for the WELFake fake-news dataset:

1. Task 1 — Initial data audit and quality assessment
2. Task 2 — Missing-value handling and data cleaning strategy
3. Task 3 — Duplicate detection and removal using text normalization, TF-IDF, and cosine similarity
4. Task 4 — Statistical text analysis comparing real and fake news distributions

Together they form a complete preprocessing workflow for preparing the dataset before modeling.

## Dataset
The project uses the WELFake Fake News Classification Dataset from Kaggle.

- Source: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
- Rows: 72,134
- Columns: 4 (`Unnamed: 0`, `title`, `text`, `label`)
- Target: binary label (`0` = Fake, `1` = Real)

> The raw `WELFake_Dataset.csv` file is not included in this repository because of its size. Download it from the Kaggle link above and place it in the corresponding notebook folder before running the notebooks.

## Project Structure
```
Task/
├── .venv/
├── requirements.txt
├── data/
│   ├── WELFake_Dataset.csv
│   ├── data_cleaned_task2.csv
│   └── data_cleaned_task3.csv
├── notebook/
│   ├── README.md
│   └── combined_project_pipeline.ipynb
├── results/
│   ├── audit_summary.csv
│   ├── duplicate_summary.csv
│   ├── missing_value_analysis.csv
│   ├── task4_distribution_boxplots.png
│   ├── task4_distribution_histograms.png
│   ├── task4_text_statistics.csv
│   └── task4_text_summary_by_label.csv
└── ...
```

## Task 1 — Initial Data Audit

### Objective
Perform a comprehensive initial audit of the dataset before building any machine learning model. This includes examining structure, identifying data quality issues, checking missing values, duplicates, inconsistencies, and text quality, and recommending suitable data types without cleaning or modeling.

### Key Findings
- The dataset has 72,134 rows and 4 columns.
- `Unnamed: 0` is a redundant row-index column.
- Missing values are concentrated in `title` and `text`.
- There are no fully duplicated rows, but many duplicate titles and duplicated article texts exist.
- The label column is clean and binary.
- The classes are nearly balanced.
- Some article texts are extremely short or empty and should be handled carefully.
- `title` and `text` should be treated as string data, while `label` should be cast appropriately for modeling.

### Notebook
- `notebook/combined_project_pipeline.ipynb` (contains the project workflow and audit section)

### Output Files
- `results/missing_value_analysis.csv`
- `results/audit_summary.csv`

## Task 2 — Missing-Value Handling

### Objective
This task focuses only on missing-value handling, using the findings from Task 1 and avoiding a repeat of the audit process.

### Final Strategy
- Keep `title` and `text` as text fields.
- Do not apply numerical imputation to text columns.
- Fill missing text values with empty strings to preserve row integrity.
- Remove rows with missing `label` values.

### Notebook
- `notebook/combined_project_pipeline.ipynb` (Task 2 section)

### Output Files
- `data/data_cleaned_task2.csv`
- `results/missing_value_analysis.csv`

## Task 3 — Duplicate Detection and Removal

### Objective
This task uses the cleaned dataset from Tasks 1 and 2 to detect and remove exact duplicates and near-duplicates without repeating earlier audit or missing-value work.

### Final Pipeline
- Normalize text by lowercasing and removing punctuation and repeated whitespace.
- Detect exact duplicates using normalized `title` + `text` pairs.
- Build a TF-IDF matrix from the combined article text.
- Compute cosine similarity between records.
- Flag near-duplicates using a similarity threshold of 0.95.
- Remove confirmed duplicates and near-duplicates while keeping the first valid record.

### Verified Results
- Rows before duplicate removal: 72,134
- Exact duplicate rows: 8,745
- Near-duplicate rows: 258
- Rows after duplicate removal: 63,131
- Total rows removed: 9,003

### Notebook
- `notebook/combined_project_pipeline.ipynb` (Task 3 section)

### Output Files
- `data/data_cleaned_task3.csv`
- `results/duplicate_summary.csv`

## Task 4 — Statistical Text Analysis

### Objective
Measure and compare text distributions for fake news (`label = 0`) and real news (`label = 1`) using the raw WELFake dataset. The analysis covers word count, non-whitespace character count, stopword frequency, total text length including spaces, and stopword ratio. For each measure, the notebook reports the mean, median, standard deviation, minimum, and maximum.

This task is descriptive only. It does not impute missing values, remove duplicates, balance classes, or train a machine-learning model. Missing article text is treated as an empty string only while calculating text statistics.

### Visualizations and Results
- Overlaid histograms compare fake and real distributions, capped at the overall 95th percentile for readability.
- Boxplots compare the central tendency and spread without displaying extreme outliers.
- `results/task4_text_statistics.csv` contains article-level measurements.
- `results/task4_text_summary_by_label.csv` contains grouped descriptive statistics.
- `results/task4_distribution_histograms.png` and `results/task4_distribution_boxplots.png` contain the comparison plots.

### Notebook
- `notebook/combined_project_pipeline.ipynb`, under the `Task 4: Statistical Text Analysis` section.

### Conclusion
The Task 4 notebook prints the conclusion from the actual raw-dataset medians after execution. The distributions should be interpreted as overlapping descriptive signals rather than as a standalone fake-news classifier: differences in article length, character count, word count, or stopword use do not by themselves establish whether an article is real or fake.

## Setup
Install the dependencies for the entire project:

```bash
pip install -r requirements.txt
```

For Task 2 specifically, the same project dependency list is used, and the notebook can be run from its folder after installation.

## Tools and Technologies
- Python 3
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook
- scikit-learn
- NLTK

## How to Run the Project
1. Download the raw dataset from the Kaggle link above and place it in `data/WELFake_Dataset.csv`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Open `notebook/combined_project_pipeline.ipynb`.
4. Run the notebook cells in order to execute the full workflow: audit, missing-value handling, duplicate removal, and Task 4 statistical analysis.
5. Review the generated outputs in `results/` and the cleaned datasets in `data/`.

## Conclusion
The WELFake dataset contains important data-quality issues, including missing text values, repeated article records, and near-duplicate headlines and articles. The workflow now also includes a raw-dataset statistical comparison of fake and real news text, giving a reproducible descriptive baseline before any downstream modeling.
