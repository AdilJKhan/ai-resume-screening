import os
import PyPDF2
import docx2txt
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    return ""

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def match_resumes(job_description, files):
    resume_texts = []
    resume_names = []

    for file in files:
        text = extract_text(file)
        resume_texts.append(text)
        resume_names.append(os.path.basename(file.name))

    # Combine job + resumes
    documents = [job_description] + resume_texts

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(documents)

    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(job_vector, resume_vectors)[0]

    # Get top 3 indices
    top_indices = similarities.argsort()[-3:][::-1]

    results = []
    for idx in top_indices:
        results.append(
            f"{resume_names[idx]} — Score: {similarities[idx]:.2f}"
        )

    return "\n".join(results)



interface = gr.Interface(
    fn=match_resumes,
    inputs=[
        gr.Textbox(label="Job Description"),
        gr.File(file_count="multiple",type="filepath" ,label="Upload Resumes"),
    ],
    outputs=gr.Textbox(label="Top Matching Resumes"),
    title="AI Resume Screening System (Demo Version)",
    description="Lightweight demo version using TF-IDF similarity."
)

interface.launch()
