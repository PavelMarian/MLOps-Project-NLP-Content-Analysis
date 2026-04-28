# MLOps-Project-NLP-Content-Analysis

## Toxicity Moderation System

A machine learning system for detecting toxic comments and moderating harmful speech in text.

This project uses a fine-tuned **RuBERT** model to classify Russian text into toxicity-related categories and provides a simplified user interface for quick moderation.

### Features

- Toxicity classification for Russian text.
- Simple and user graphical interface.
- PyTorch-based model inference.
- Hugging Face Transformers integration.
- Modular project structure for training, evaluation, and deployment.

### Libraries

- **PyTorch** — deep learning framework.
- **Hugging Face Transformers** — model loading, tokenization, and inference.
- **Streamlit** — simplified user interface.
- **scikit-learn** — metrics and evaluation utilities.
- **pandas** — data processing.

### Model

**RuBERT** is a Russian-language version of BERT, adapted for NLP tasks in Russian.

The model is used for toxicity detection and moderation of user-generated content.

### Datasets

The project's model was trained and fine-tuned on the following datasets:

- **Russian Hate Speech (14k)** — binary classification.
- **Toxic Russian Comments (100k+)** — 4-class classification.
- **Russian Inappropriate Messages (150k)** — binary classification plus topic labels.

### Project Structure


### UI


### Installation


### Usage

#### Run the app

#### Run inference from Python

### Evaluation

### Model Artifacts

Fine-tuned model weights are stored in Git LFS.  

### Ethics and Limitations

This system is intended to assist moderation, not replace human review.

- Toxicity classification can produce false positives and false negatives.
- Context, irony, quotation, and slang may affect accuracy.
- Human oversight is recommended for real moderation pipelines.
