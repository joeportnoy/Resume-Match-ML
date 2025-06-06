from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util #loads a pretrained model that converts text into numerical embeddings (vectors).
# util computes cosine similarity between those vectors (i.e., how similar two texts are).
import torch
import os
import fitz  # reads PDF files (PyMuPDF library).
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for servers
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer # turns text into word frequency vectors to find overlapping keywords.
import nltk #handles stopwords like “the”, “and” so they don’t pollute results.
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 
# Initializes the Flask app.
# Limits file uploads to 5MB and saves them in the uploads/ folder.

# Load SBERT model
device = 'cpu'
model = SentenceTransformer('https://huggingface.co/joeportnoy/resume-match-ml', device=device)
# Loads the pretrained Sentence-BERT model, which converts full text (like resumes) into a numerical embedding representing its meaning.
# This model was trained to understand semantic similarity between sentences.

# Reads all the text from the uploaded PDF using PyMuPDF and joins it into a single string.
def read_pdf(file_path):
    doc = fitz.open(file_path)
    return " ".join([page.get_text() for page in doc])

# Creates a donut-style pie chart showing how close the resume matches the job description.
# Saves it to static/donut.png for display in the browser.


def create_donut_chart(score, out_path="static/donut.png"):
    plt.figure(figsize=(4, 4))
    sizes = [score, 100 - score]
    colors = ['#4CAF50', '#dddddd']
    labels = [f'{score}% Match', '']

    wedges, _ = plt.pie(sizes, colors=colors, startangle=90, labels=labels)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title("Resume Match Score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

#This defines the main web page (/) and allows both loading the form (GET) and submitting it (POST).

@app.route('/', methods=['GET', 'POST'])
def index():
    match_score = None
    error = None
    top_keywords = None  # <- define this early to avoid "referenced before assignment"

    if request.method == 'POST':
        job_desc = request.form['job_description']
        file = request.files['resume']

        if not file.filename.endswith('.pdf'):
            error = "Only PDF files are supported. Please upload a .pdf file."
        else:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            resume_text = read_pdf(filepath)

            # Converts both resume and job description into dense vector embeddings using the SBERT model.
            # Calculates cosine similarity, which gives a number between 0 and 1.
            # Multiplies by 100 to show it as a percentage match.
            resume_emb = model.encode(resume_text, convert_to_tensor=True)
            jd_emb = model.encode(job_desc, convert_to_tensor=True)
            similarity = util.cos_sim(resume_emb, jd_emb)
            match_score = round(float(similarity.item()) * 100, 2)

            # Generate donut chart and extract top keywords
            create_donut_chart(match_score)
            top_keywords = extract_top_keywords(resume_text, job_desc)
            #Uses TF-IDF (term frequency-inverse document frequency) to identify the most significant words in both texts.
            #Finds keywords that are important in both resume and JD.
            #Returns the top N overlapping words

    return render_template('index.html', match_score=match_score, error=error, keywords=top_keywords)



def extract_top_keywords(resume_text, jd_text, top_n=5):
    # Combine both for vectorization
    corpus = [resume_text, jd_text]

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # Get vectors
    resume_vec = tfidf_matrix[0].toarray()[0]
    jd_vec = tfidf_matrix[1].toarray()[0]

    # Multiply vectors element-wise to find shared importance
    combined = resume_vec * jd_vec

    # Get top N overlapping keywords
    top_indices = combined.argsort()[-top_n:][::-1]
    keywords = [(feature_names[i], round(combined[i], 3)) for i in top_indices if combined[i] > 0]

    return [kw[0] for kw in keywords]


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
