# AI Resume Screening System (Python, Flask, Sentence-Transformers)

This project is an end-to-end AI-powered resume screening system built using
transformer-based semantic embeddings in Python.

It demonstrates applied NLP in a real-world HR Tech scenario:
document parsing, embedding generation, similarity scoring,
multi-file processing, and web-based interaction.

---

## Features

- Semantic similarity matching using Sentence-Transformers
- PDF, DOCX, and TXT resume support
- Cosine similarity ranking
- Batch resume processing
- Flask-based web interface
- Recruiter-ready demo dataset

---

## Model Overview

- **Embedding Model**: `all-MiniLM-L6-v2`
- **Embedding Framework**: Sentence-Transformers
- **Similarity Metric**: Cosine Similarity
- **Matching Strategy**:
  - Convert job description into embedding
  - Convert each resume into embedding
  - Compute similarity between job and resumes
  - Rank resumes by descending similarity score

Unlike traditional keyword-based matching (TF-IDF),
this system captures semantic meaning and contextual similarity.

---

## Project Structure

```
ai-resume-screening/
├── demo_data/
│   ├── job_descriptions/
│   └── resumes/
├── static/
├── templates/
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python main.py
```

Then open in browser:

```
http://127.0.0.1:5000/
```

---

## Live Demo

A lightweight TF-IDF demo version is deployed on Hugging Face Spaces:

https://huggingface.co/spaces/AdilJKhan/ai-resume-screening

> Note: The live demo uses TF-IDF due to free-tier memory constraints.
> The full transformer-based semantic version is available in this repository.

---

## Technologies Used

- Python
- Flask
- Sentence-Transformers
- Hugging Face Transformers
- scikit-learn
- PyPDF2
- docx2txt

---

## Project Purpose

This project simulates an AI-assisted Applicant Tracking System (ATS) component that:

- Reduces manual resume screening time
- Improves candidate-job matching accuracy
- Demonstrates practical NLP deployment skills

---

## Author

Adil Khan
