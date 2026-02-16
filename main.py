import os
import PyPDF2
import docx2txt
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# TEXT EXTRACTION FUNCTIONS
# -----------------------------

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    return ""

@app.route("/", methods=["GET", "POST"])
def matcher():

    if request.method == "POST":

        job_description = request.form.get("job_description")
        resume_files = request.files.getlist("resumes")

        if not job_description or not resume_files:
            return render_template("index.html", message="Please upload resumes and enter job description.")

        saved_files = []
        resume_texts = []

        for resume_file in resume_files:
            if resume_file.filename == "":
                continue

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
            resume_file.save(file_path)

            saved_files.append(resume_file.filename)
            resume_texts.append(extract_text(file_path))

        if not resume_texts:
            return render_template("index.html", message="No valid resumes uploaded.")

        # Generate embeddings
        job_embedding = model.encode(job_description, convert_to_tensor=True)
        resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)

        # Compute cosine similarities
        similarity_scores = util.cos_sim(job_embedding, resume_embeddings)[0]

        # Convert tensor to list
        similarity_scores = similarity_scores.cpu().tolist()

        # Get top 3 matches
        top_indices = sorted(range(len(similarity_scores)),
                     key=lambda i: similarity_scores[i],
                     reverse=True)[:3]

        top_resumes = [saved_files[i] for i in top_indices]
        top_scores = [round(similarity_scores[i] * 100, 2) for i in top_indices]


        return render_template(
            "index.html",
            message="Top Matching Resumes:",
            top_resumes=top_resumes,
            similarity_score=top_scores
        )

    return render_template("index.html")


# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)