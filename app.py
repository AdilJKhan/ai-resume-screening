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


def match_resumes(job_description, resume_files):
    resumes_text = []

    for file in resume_files:
        text = extract_text(file)
        resumes_text.append(text)

    vectorizer = TfidfVectorizer().fit_transform(
        [job_description] + resumes_text
    )

    vectors = vectorizer.toarray()
    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    similarities = cosine_similarity([job_vector], resume_vectors)[0]

    results = []
    for i, score in enumerate(similarities):
        results.append(
            f"{resume_files[i].name} — Score: {round(score * 100, 2)}"
        )

    results.sort(reverse=True)
    return "\n".join(results[:3])


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
