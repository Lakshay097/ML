from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer,SentenceTransformerTrainingArguments
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import streamlit as st
from transformers import pipeline

data = pd.read_csv("data/job_title_des.csv", usecols=["Job Title", "Job Description"])
data["text"] = data["Job Title"] + " - " + data["Job Description"]
job_texts = data["text"].to_list()

Embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(Embedding_model)

"""job_embb = model.encode(job_texts, convert_to_numpy=True, normalize_embeddings=True)

np.save("models/job_embeddings.npy", job_embb)"""

job_embb = np.load("models/job_embeddings.npy")

@st.cache_resource
def ner_func():
    return pipeline(
        task="ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )

def extract_skills(inputs):
    ner = ner_func()
    entities = ner(inputs)
    print("ENTITIES:", entities)
    skills = [
        s["word"].strip().lower()
        for s in entities
        if s.get("score", 0) >= 0.3 and len(s["word"].strip()) > 2
    ]
    seen, res = set() , []
    for i in skills:
        if i not in seen:
            seen.add(i)
            res.append(i)

    return res

def func(inputs, top_k=5):
    emb = model.encode([inputs], convert_to_numpy=True, normalize_embeddings=True)
    similarity = cosine_similarity(emb, job_embb)[0]

    top_idx = np.argsort(similarity)[-top_k:][::-1]
    resume_skills = set(extract_skills(inputs))

    results = []
    for i in top_idx:
        job_text = data.iloc[i]["text"]
        job_skills = set(extract_skills(job_text))

        overlap = resume_skills & job_skills
        missing = job_skills - resume_skills
        skill_match_pct = float(len(overlap) / len(job_skills)) if job_skills else 0.0

        results.append(
            {
                "job_index": int(i),
                "job_title": data.iloc[i]["Job Title"],
                "similarity": float(similarity[i]),
                "resume_skills": sorted(list(resume_skills)),
                "job_skills": sorted(list(job_skills)),
                "overlap_skills": sorted(list(overlap)),
                "missing_skills": sorted(list(missing)),
                "skill_match_pct": skill_match_pct,
            }
        )
    return results


st.title("Resume–Job Matcher (Embeddings)")

resume_text = st.text_area("Paste your resume text", height=250)
slider = st.slider(label="Top Jobs Title",min_value=1,max_value=10)

if st.button("Match") and resume_text.strip():
    results = func(resume_text, top_k=slider)
    for r in results:
        st.subheader(f"{r['job_title']}")
        st.write(f"Similarity: {r['similarity']:.3f}")
        st.write(f"Skill match: {r['skill_match_pct']*100:.1f}%")
        st.write("Overlap skills:", ", ".join(r["overlap_skills"]) or "-")
        st.write("Missing skills:", ", ".join(r["missing_skills"]) or "-")
        st.markdown("---")