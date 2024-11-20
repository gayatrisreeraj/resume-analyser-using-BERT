import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PyPDF2 import PdfReader
import zipfile
import os
import tempfile
from collections import defaultdict
from bert_score import score
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

category_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=25)
section_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

category_model.load_state_dict(torch.load('fine-tuned_BERT_resume-classification.pt', map_location=torch.device('cpu')))
section_model.load_state_dict(torch.load('fine-tuned_BERT_resume-sections.pt', map_location=torch.device('cpu')))

category_model.eval()
section_model.eval()

category_dict = {
    0: 'Data Science', 1: 'HR', 2: 'Advocate', 3: 'Arts', 4: 'Web Designing',
    5: 'Mechanical Engineer', 6: 'Sales', 7: 'Health and fitness', 8: 'Civil Engineer',
    9: 'Java Developer', 10: 'Business Analyst', 11: 'SAP Developer', 
    12: 'Automation Testing', 13: 'Electrical Engineering', 14: 'Operations Manager',
    15: 'Python Developer', 16: 'DevOps Engineer', 17: 'Network Security Engineer',
    18: 'PMO', 19: 'Database', 20: 'Hadoop', 21: 'ETL Developer', 
    22: 'DotNet Developer', 23: 'Blockchain', 24: 'Testing'
}

sections = ['others', 'experience', 'knowledge', 'project', 'education']
category_to_idx = {section: idx for idx, section in enumerate(sections)}
idx_to_category = {idx: section for section, idx in category_to_idx.items()}

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_line(line):
    line = re.sub(r'[^\w\s\t]', '', line)
    line = re.sub(r'[\u2013\u2014\u2022\u25CF]', ' ', line)
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def predict_category(text, model, tokenizer, category_to_idx, idx_to_category, device):
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=-1)

    predicted_category = idx_to_category[prediction.item()]
    return predicted_category

def analyse_resume_sections(resume_text):
    sectioned_lines = defaultdict(list)

    lines = resume_text.split("\n")
    for line in lines:
        line = clean_line(line)
        if line.strip():
            predicted_section = predict_category(line, section_model, tokenizer, category_to_idx, idx_to_category, device)
            sectioned_lines[predicted_section].append(line)

    return sectioned_lines

def analyse_single_resume(uploaded_resume):
    resume_text = extract_text_from_pdf(uploaded_resume)
    
    predicted_category = predict_job_category(resume_text)
    st.write(f"### Predicted Job Category: {predicted_category}")
    
    sectioned_lines = analyse_resume_sections(resume_text)
    
    st.write("### Resume Sections Analysis")
    for section, lines in sectioned_lines.items():
        st.write(f"**{section.capitalize()}**:")
        for line in lines:
            st.write(f"- {line}")

def calculate_similarity(resume_text, job_desc_text):
    P, R, F1 = score([resume_text], [job_desc_text], lang="en")
    return F1.mean().item()

def predict_job_category(resume_text):
    encoding = tokenizer(
        resume_text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = category_model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=-1)

    predicted_category = category_dict[prediction.item()]
    return predicted_category

def find_lacking_words(resume_text, job_desc_text):
    resume_words = set(resume_text.lower().split())
    job_desc_words = set(job_desc_text.lower().split())
    
    lacking_words = job_desc_words - resume_words
    
    return list(lacking_words)

def rank_resumes_by_similarity(zip_file, job_desc_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        job_desc_text = extract_text_from_pdf(job_desc_file)

        resume_scores = []

        for file_name in os.listdir(temp_dir):
            if file_name.endswith(".pdf"):
                resume_path = os.path.join(temp_dir, file_name)
                resume_text = extract_text_from_pdf(resume_path)

                similarity_score = calculate_similarity(resume_text, job_desc_text)

                job_category = predict_job_category(resume_text)

                lacking_words = find_lacking_words(resume_text, job_desc_text)

                resume_scores.append({
                    'resume': file_name,
                    'similarity': similarity_score,
                    'category': job_category,
                    'lacking_words': lacking_words[:20]
                })

        ranked_resumes = sorted(resume_scores, key=lambda x: x['similarity'], reverse=True)

        st.write("### Ranked Resumes Based on Similarity to Job Description")
        for rank, resume in enumerate(ranked_resumes, 1):
            st.write(f"**Rank {rank}: {resume['resume']}**")
            st.write(f"Similarity: {resume['similarity']:.4f}")
            st.write(f"Predicted Category: {resume['category']}")
            st.write(f"Lacking Words: {', '.join(resume['lacking_words'])}")
            st.write("---")

def main():
    st.title("Resume Analyser and Ranker")

    # Option 1: Single Resume Section Analysis
    option = st.radio("Choose Option", ["Analyse Resume", "Rank Resumes"])

    if option == "Analyse Resume":
        uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        if uploaded_resume:
            analyse_single_resume(uploaded_resume)

    # Option 2: Multiple Resume Ranking
    if option == "Rank Resumes":
        zip_file = st.file_uploader("Upload Resumes (ZIP)", type=["zip"])
        job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

        if zip_file and job_desc_file:
            rank_resumes_by_similarity(zip_file, job_desc_file)

if __name__ == "__main__":
    main()