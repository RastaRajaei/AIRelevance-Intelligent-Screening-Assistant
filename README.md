# AIRelevance: Intelligent Screening Assistant for Systematic Review of Forest Inventory Literature

**AIRelevance** -AI-driven Relevance- is an AI-powered tool designed to assist researchers in the initial screening phase of systematic reviews, focusing on forest inventory studies using UAV-RGB imagery and Convolutional Neural Networks (CNNs). It automates the classification of research articles as "related" or "unrelated" based on their relevance to tasks such as tree delineation, species classification, individual tree detection, instance segmentation, and vitality assessment.

## Overview
This tool processes BibTeX (`.bib`) files containing article metadata (title and abstract) and uses text embeddings to compute semantic similarity with a predefined query. Articles are categorized based on a similarity threshold, streamlining the screening process for systematic reviews in forest inventory research.

## Models Used
The tool leverages the following pre-trained models from Hugging Face:

- **google/embeddinggemma-300m**  
  - Source: [Hugging Face](https://huggingface.co/google/embeddinggemma-300m)  
  - License: Apache 2.0  
  - Description: A 300M parameter open embedding model from Google, built from Gemma 3 with T5Gemma initialization. It produces 768-dimensional vector representations (with Matryoshka Representation Learning for smaller sizes: 512, 256, or 128) and is optimized for text classification, clustering, and semantic similarity tasks. Suitable for resource-constrained environments like laptops or mobile devices.  
  - Citation: Google DeepMind, [EmbeddingGemma on Kaggle](https://www.kaggle.com/models/google/embedding-gemma), [Vertex Model Garden](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/embedding-gemma).

- **sentence-transformers/all-MiniLM-L6-v2**  
  - Source: [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
  - License: Apache 2.0  
  - Description: A lightweight model for generating text embeddings, used as a fallback or alternative for faster processing on limited hardware.  

## Prerequisites
- Python 3.8+
- Google Colab (recommended) or a local environment with GPU support
- Google Drive for storing `.bib` files
- Hugging Face account and access token for `google/embeddinggemma-300m` (optional for `all-MiniLM-L6-v2`)

## Installation
1. Install required libraries:
   ```bash
   pip install bibtexparser python-docx transformers accelerate sentencepiece

Mount your Google Drive in Colab to access .bib files.
(Optional) Set up a Hugging Face access token for google/embeddinggemma-300m:

Create a Fine-grained token with "Read access to contents of all public gated repos you can access" at Hugging Face Settings.
Set the token as an environment variable:
pythonimport os
os.environ["HF_TOKEN"] = "your_hf_token"
Or use !huggingface-cli login in Colab.



Usage

Place your .bib files in a Google Drive folder (e.g., /content/drive/MyDrive/Research/Bibtex_Folder).
Update the folder_path variable in the script to point to your folder.
Choose a model in the load_model function:

For google/embeddinggemma-300m: Requires HF token, higher accuracy for complex tasks.
For sentence-transformers/all-MiniLM-L6-v2: No token needed, faster for large datasets.


Run the script in Colab. The script will:

Load .bib files and extract title/abstract.
Generate embeddings using the selected model.
Compute cosine similarity with the query.
Save results to:

classified_articles.csv: Contains article details (ID, title, abstract), similarity scores, source file, and category (related/unrelated).
related.bib: Articles classified as relevant (score ≥ 0.50).
unrelated.bib: Articles classified as non-relevant (score < 0.50).



Code
The main script is available in classify_articles.py. To switch between models, uncomment the desired model in the load_model function:

For EmbeddingGemma: model_name = "google/embeddinggemma-300m"
For MiniLM: model_name = "sentence-transformers/all-MiniLM-L6-v2"

License

Project License: MIT (see LICENSE file for details).
Model Licenses:

google/embeddinggemma-300m: Apache 2.0 (see Hugging Face Model Card).
sentence-transformers/all-MiniLM-L6-v2: Apache 2.0 (see Hugging Face Model Card).


Ensure compliance with each model's license when using or redistributing.

Notes

The similarity threshold (default: 0.50) can be adjusted for higher recall (e.g., 0.40) or precision (e.g., 0.65). Lower thresholds are recommended for systematic reviews to minimize missing relevant articles.
Do not share your Hugging Face token publicly. Use environment variables or secure storage.
Input .bib files should contain valid BibTeX entries with title and abstract fields for best results.
For large datasets, consider saving embeddings (np.save) to avoid recomputation.
Manual review of the related.bib output is recommended to ensure no relevant articles are missed.

Citation
If you use this code or models in your research, please cite:

Models:

Google DeepMind, EmbeddingGemma, Hugging Face, Kaggle.
Sentence Transformers, all-MiniLM-L6-v2, Hugging Face.


This Repository: https://github.com/RastaRajaei/AIRelevance-Intelligent-Screening-Assistant.git

Acknowledgments
This project was developed to support systematic reviews in forest inventory using UAV-RGB imagery and CNNs. Thanks to Google DeepMind and the Sentence Transformers team for providing open-access models under the Apache 2.0 license.