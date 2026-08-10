import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice

MODEL_NAME = "gauravjhaiitm/deberta-mcq-solver"
OPTIONS = list('ABCDE')
MAX_LEN = 256

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

# Added loading spinner so users know the app is booting up
with st.spinner("Loading DeBERTa v3 weights..."):
    tokenizer, model = load_model()

st.title("MCQ Solver — DeBERTa v3")
st.write("Enter a question and 5 options. Model will predict top 3 answers.")

# Initialize session state
for field in ['prompt', 'A', 'B', 'C', 'D', 'E']:
    if field not in st.session_state:
        st.session_state[field] = ''

# Clear fields button

def clear_fields():
    for field in ['prompt', 'A', 'B', 'C', 'D', 'E']:
        st.session_state[field] = ''

st.button("Clear Fields", on_click=clear_fields)

with st.form("mcq_form"):
    prompt = st.text_area("Question / Prompt", key='prompt')
    A = st.text_input("Option A", key='A')
    B = st.text_input("Option B", key='B')
    C = st.text_input("Option C", key='C')
    D = st.text_input("Option D", key='D')
    E = st.text_input("Option E", key='E')
    
    submitted = st.form_submit_button("Predict")

if submitted:
    if prompt.strip() and all([A.strip(), B.strip(), C.strip(), D.strip(), E.strip()]):
        options = [A, B, C, D, E]
        
        encoded  = tokenizer(
            [prompt] * 5, options,
            truncation=True, max_length=MAX_LEN,
            padding='max_length', return_tensors='pt'
        )
        
        inputs = {k: v.unsqueeze(0) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        ranked = np.argsort(-probs)
        
        st.success("Top 3 Predictions:")
        for i in range(3):
            rank_idx = ranked[i]
            st.write(f"**Rank {i+1}: Option {OPTIONS[rank_idx]}** ({options[rank_idx]}) — {probs[rank_idx]:.2%} confidence")
    else:
        st.warning("Please fill in all fields.")