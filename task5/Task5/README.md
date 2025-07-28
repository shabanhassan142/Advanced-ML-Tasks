# Auto Tagging Support Tickets Using LLM (Task 5)

## 1. Objective of the Task
The objective of this task is to automatically tag support tickets into predefined categories using a large language model (LLM). The approach leverages zero-shot learning, allowing the model to classify tickets without any additional training or fine-tuning, making it suitable for rapid deployment in scenarios with limited labeled data.

## 2. Methodology / Approach
- **Data Loading:**
  - The support ticket dataset (`all_tickets_processed_improved_v3.csv`) is loaded using pandas. Each ticket contains a free-text description (`Document`) and a category label (`Topic_group`).
- **Label Extraction:**
  - All unique ticket categories (tags) are extracted from the dataset to serve as candidate labels for classification.
- **Zero-Shot Classification:**
  - The Hugging Face `transformers` library is used to initialize a zero-shot classification pipeline with the `facebook/bart-large-mnli` model.
  - For each ticket, the model predicts the most relevant tags from the list of possible categories, outputting the top 3 most probable tags.
- **Evaluation:**
  - A random sample of 500 tickets is selected for evaluation.
  - For each ticket, the code checks if the true label is among the top 3 predicted tags (top-3 accuracy).
  - The results, including the ticket text, true label, predicted tags, and correctness, are displayed for review.

## 3. Key Results or Observations
- **Model Used:** facebook/bart-large-mnli (zero-shot classification)
- **Number of Categories:** 8 unique ticket categories
- **Top-3 Accuracy:** The zero-shot top-3 accuracy on the sample is **28%** (i.e., in 28% of cases, the true label is among the top 3 predictions).
- **Insights:**
  - Zero-shot LLMs can provide reasonable predictions even without fine-tuning, but accuracy may be limited depending on the complexity and overlap of categories.
  - The approach is fast to implement and does not require labeled training data, making it ideal for prototyping or low-resource settings.

## 4. Libraries Used and Why
- **pandas**: Used for loading, manipulating, and analyzing the support ticket dataset (CSV file). It provides powerful data structures for handling tabular data.
- **transformers**: From Hugging Face, this library provides access to state-of-the-art large language models and pipelines, including zero-shot classification with models like `facebook/bart-large-mnli`.
- **tqdm**: Used for displaying progress bars during batch processing, making it easier to monitor the progress of operations like applying the classification function to many tickets.
- **sentence-transformers** (installed, but not directly used in the main code): Useful for advanced sentence embeddings and semantic search tasks.
- **tf-keras** (installed, but not directly used in the main code): A deep learning library for building and training neural networks, included for potential future extensions.

---

