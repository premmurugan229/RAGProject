# app.py
import streamlit as st
from graph import app

st.title("Multimodal Agentic RAG")
uploaded_file = st.file_uploader("Upload text/image/CSV")
query = st.text_input("Query")

if st.button("Run"):
    modality = detect_modality(uploaded_file)
    input_data = preprocess_image(Image.open(uploaded_file)) if "image" in modality else uploaded_file.read()
    result = app.invoke({"input": input_data, "query": query})
    st.write(result["answer"])
    st.json({"confidence": result["confidence"], "provenance": result["sources"]})
