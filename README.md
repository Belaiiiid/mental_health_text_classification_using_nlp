# Mental Health Text Classification Using NLP and Machine Learning

## Project Overview

This project explores the use of Natural Language Processing (NLP) and Machine Learning to classify text related to mental health.

The objective is to analyze textual statements and predict their corresponding mental health category using different text representation techniques and classification models.

The project compares traditional Machine Learning approaches based on **Document-Term Matrix (DTM)** and **TF-IDF** with a fine-tuned **BERT** model to evaluate the impact of different text representations on classification performance.

> This project is intended for educational and research purposes. The models are not designed to provide clinical diagnoses or replace professional mental health assessments.

## Objectives

- Explore and preprocess a mental health text dataset.
- Convert unstructured text into numerical representations.
- Implement and compare classical Machine Learning models.
- Fine-tune a pre-trained Transformer model for text classification.
- Evaluate model performance using classification metrics.
- Analyze classification errors, limitations, and potential improvements.

## Dataset

The project uses `Combined Data.csv`, containing textual statements and their corresponding mental health categories.

The main columns used are:

| Column | Description |
|---|---|
| `statement` | Original textual statement |
| `statement_clean` | Preprocessed text used for modeling |
| `status` | Original target category |
| `status_encoded` | Numerically encoded target category |

The dataset is explored and preprocessed before being used for model training.

## Project Workflow

```text
Dataset
   |
   v
Exploratory Data Analysis
   |
   v
Text Preprocessing
   |
   v
Target Encoding
   |
   v
Train / Test Split
   |
   +-----------------------------+
   |                             |
   v                             v
Classical NLP                 BERT
   |                             |
   v                             v
DTM and TF-IDF              Tokenization
   |                             |
   v                             v
Random Oversampling         Fine-tuning
   |                             |
   v                             |
Logistic Regression              |
Random Forest                    |
   |                             |
   +--------------+--------------+
                  |
                  v
          Model Evaluation
                  |
                  v
             Error Analysis
                  |
                  v
         Critical Discussion
```

## Methodology

### 1. Exploratory Data Analysis

The dataset is explored to understand its structure and characteristics, including:

- Dataset dimensions and column types
- Missing values and duplicated records
- Distribution of mental health categories
- Text length and general characteristics of statements

This step helps identify data quality issues and potential class imbalance before modeling.

### 2. Text Preprocessing

Text preprocessing prepares the statements for NLP modeling.

The workflow includes cleaning the textual data and creating a dedicated `statement_clean` column for the classification task.

The cleaned statements are then used as input for the classical NLP models and BERT.

### 3. Target Encoding

The categorical target variable `status` is converted into numerical labels using `LabelEncoder`.

This transformation makes the target compatible with the classification algorithms.

### 4. Train-Test Split

The dataset is divided into training and testing subsets using an 80/20 split.

For BERT, the training subset is further divided into:
- 85% for model training
- 15% for validation

Stratified splitting is used for the BERT training-validation division to preserve the class distribution.

The test subset is kept separate for final evaluation.

### 5. Text Representation: DTM and TF-IDF

Since traditional Machine Learning models cannot directly process raw text, textual statements are transformed into numerical feature matrices.

#### Document-Term Matrix (DTM)

DTM represents each document using the frequency of terms appearing in it.

The `CountVectorizer` implementation uses:

- Unigrams and bigrams: `ngram_range=(1, 2)`
- English stop-word removal
- Maximum document frequency: `max_df=0.7`
- Maximum vocabulary size: `max_features=50,000`

Each row represents a document, each column represents a term, and each value represents its frequency in the document.

#### TF-IDF

TF-IDF assigns a weight to each term based on its frequency within a document and its rarity across the training corpus.

It reduces the relative importance of terms that appear frequently across many documents and can emphasize more informative terms.

The implementation uses:

- Unigrams and bigrams
- Maximum vocabulary size: `50,000`
- Default scikit-learn TF-IDF weighting and normalization

Both vectorizers are fitted exclusively on the training data and then applied to the test data to avoid data leakage.

### 6. Class Imbalance Handling

Random Over-Sampling is applied to the training data used by the classical Machine Learning models.

This technique increases the representation of minority classes by randomly duplicating existing samples.

Oversampling is restricted to the training data to prevent information leakage into the test set.

### 7. Classical Machine Learning Models

Two classification algorithms are implemented and evaluated using both DTM and TF-IDF representations.

#### Logistic Regression

Logistic Regression is used as a linear classification model. It learns the relationship between the numerical text features and the target categories.

It provides a classical baseline for evaluating text classification performance.

#### Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to produce classification predictions.

It is evaluated with both DTM and TF-IDF to compare its performance across the two text representations.

### 8. BERT Fine-Tuning

A pre-trained `bert-base-uncased` model is fine-tuned for sequence classification using the Hugging Face Transformers library.

Unlike DTM and TF-IDF, BERT generates contextual representations of text, allowing the model to learn relationships between words based on their surrounding context.

#### BERT pipeline

1. Load the pre-trained BERT tokenizer.
2. Split the training data into training and validation sets.
3. Tokenize the statements with truncation and padding.
4. Convert the data into Hugging Face datasets.
5. Load BERT with a classification head adapted to the number of target classes.
6. Fine-tune the model on the training data.
7. Evaluate the model on the validation set after each epoch.
8. Select the best checkpoint according to the validation F1-score.
9. Evaluate the selected model on the independent test set.

#### Main training parameters

| Parameter | Value |
|---|---:|
| Pre-trained model | `bert-base-uncased` |
| Maximum sequence length | 128 |
| Learning rate | 2e-5 |
| Training batch size | 8 |
| Evaluation batch size | 16 |
| Number of epochs | 3 |
| Weight decay | 0.01 |
| Best model selection | Weighted F1-score |
| Mixed precision | FP16 when CUDA is available |

The model is fine-tuned using PyTorch and the Hugging Face `Trainer` API.

## Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy:** Overall proportion of correctly classified statements.
- **Precision:** Proportion of correct positive predictions.
- **Recall:** Proportion of actual class instances correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

Weighted precision, recall, and F1-score are used in the reported comparison to account for class support.

## Results

The following results were obtained during the experiments:

| Model | Representation | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Logistic Regression | DTM | 75.42% | 72.04% | 71.04% | 71.48% |
| Random Forest | DTM | 72.61% | 80.37% | 59.79% | 65.34% |
| Logistic Regression | TF-IDF | 76.60% | 71.41% | 74.40% | 72.70% |
| Random Forest | TF-IDF | 72.67% | 82.11% | 59.07% | 65.16% |
| BERT | Contextual | 78.87% | 75.10% | 74.86% | 74.93% |

### Results Interpretation

- BERT obtained the highest accuracy and F1-score among the evaluated models in this experiment.
- Logistic Regression with TF-IDF achieved the strongest results among the classical approaches.
- TF-IDF slightly improved Logistic Regression compared with DTM.
- Random Forest achieved relatively high precision but lower recall, indicating that it missed a larger proportion of actual instances in some classes.
- BERT's results show the potential benefit of contextual text representations for this classification task.

These results are specific to the dataset, preprocessing pipeline, model configurations, and evaluation split used in this project.

## Error Analysis

Error analysis is used to better understand the behavior and limitations of the trained models.

The analysis focuses on:

- Misclassified statements
- Confusion between target categories
- Ambiguous or context-dependent statements
- Short texts with limited information
- Potential overlap between mental health categories
- Possible annotation inconsistencies

Confusion matrices and individual misclassified examples can help identify patterns that aggregate metrics alone may not reveal.

## Limitations

- **Class imbalance:** Some categories may be more represented than others, which can affect classification performance.
- **Contextual ambiguity:** A statement may contain language associated with more than one category.
- **Dataset representativeness:** The dataset may not reflect the full diversity of real-world mental health language.
- **Interpretability:** Contextual Transformer models are more difficult to interpret than simpler models.
- **Sequence length:** BERT truncates statements exceeding the selected maximum length of 128 tokens.
- **Generalization:** Performance on the current test split does not guarantee similar performance on external datasets.
- **Clinical limitations:** Text classification predictions cannot be interpreted as medical diagnoses.

## Future Improvements

Potential directions for further work include:

- Hyperparameter optimization for classical models and BERT
- Comparison with other Transformer architectures, such as RoBERTa and DistilBERT
- Evaluation using macro F1-score and class-level metrics
- Comparison of oversampling with class-weighted learning
- Explainability methods such as SHAP, LIME, and Transformer-specific interpretation techniques
- External validation using an independent dataset
- Analysis of model robustness to spelling errors, informal language, and ambiguous statements

## Technologies Used

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **NLP:** CountVectorizer, TfidfVectorizer
- **Deep Learning:** PyTorch
- **Transformers:** Hugging Face Transformers
- **Datasets:** Hugging Face Datasets
- **Evaluation:** Scikit-learn metrics
- **Environment:** Jupyter Notebook / Google Colab

## Repository Structure

```text
mental-health-nlp-classification/
│
├── data/
│   └── Combined Data.csv
│
├── notebooks/
│   └── mental_health_nlp_classification.ipynb
│
├── results/
│   ├── model_comparison.csv
│   └── figures/
│       ├── confusion_matrix.png
│       └── ...
│
├── requirements.txt
└── README.md
```

The repository structure is a suggested organization. Include only the files and results actually available in the repository. Avoid committing sensitive data or datasets whose license does not permit redistribution.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/mental-health-nlp-classification.git
cd mental-health-nlp-classification
```

Install the required libraries:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
pip install torch transformers datasets
```

Open the notebook in Jupyter or Google Colab and execute the workflow.

For BERT fine-tuning, using a GPU-enabled environment is recommended to reduce training time.

## Conclusion

This project implements an end-to-end NLP classification workflow, from text preprocessing and numerical representation to model training and evaluation.

By comparing DTM, TF-IDF, and contextual BERT representations, the project investigates how different approaches to text representation affect classification performance.

In the reported experiment, BERT achieved an F1-score of 74.93%, compared with 72.70% for Logistic Regression using TF-IDF. The results provide a basis for further experimentation with model optimization, explainability, and external validation.

---

**Project:** Mental Health Text Classification using NLP and Machine Learning  
**Focus:** Text Classification, NLP, Machine Learning, Deep Learning, Transformers, BERT

