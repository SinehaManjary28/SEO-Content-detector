# 🧠 SEO Content Quality & Duplicate Detector

A complete machine learning pipeline that analyzes website content for **SEO quality scoring** and **duplicate content detection**.  
This project parses HTML pages, extracts meaningful text, engineers NLP-based features, detects near-duplicates, and classifies overall content quality (Low / Medium / High).

---

## 🚀 Project Overview

This assignment demonstrates an end-to-end data science workflow:
- Parse and clean HTML content
- Extract SEO-related text features
- Compute similarity to find duplicate or thin content
- Train a model to automatically score content quality
- Provide real-time analysis of any URL using `analyze_url()`

---

## 🧩 Directory Structure

seo-content-detector/
├── data/
│ ├── data.csv # Provided dataset (URLs + HTML content)
│ ├── extracted_content.csv # Parsed clean text + word counts
│ ├── features.csv # Feature-engineered dataset
│ └── duplicates.csv # Duplicate page pairs (cosine sim > 0.80)
│
├── notebooks/
│ └── seo_pipeline.ipynb # Main analysis notebook
│
├── models/
│ ├── quality_model.pkl # Trained RandomForest model
│ └── label_encoder.pkl # Encoder for quality labels
│
├── requirements.txt
└── README.md


---

## ⚙️ Setup Instructions

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/seo-content-detector
cd seo-content-detector
pip install -r requirements.txt


