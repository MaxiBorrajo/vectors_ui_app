import os
from PIL import Image
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import torch
import streamlit as st

def get_image_embedding(image: Image.Image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs[0].tolist()

def process_and_store_file(file, folder, collection, text_model, clip_model, clip_processor):
    ext = file.name.split(".")[-1].lower()
    file_path = os.path.join(folder, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())

    if ext == "txt":
        loader = TextLoader(file_path)
        docs = loader.load()
    elif ext == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext in ["jpg", "jpeg", "png"]:
        image = Image.open(file_path).convert("RGB")
        embedding = get_image_embedding(image, clip_model, clip_processor)
        collection.add(
            documents=["Imagen subida"],
            embeddings=[embedding],
            ids=[f"{file.name}_img"],
            metadatas=[{"source": file.name, "type": "image"}]
        )
        return f"🖼️ Imagen {file.name} indexada"
    else:
        return f"❌ Formato no soportado: {ext}"

    for idx, doc in enumerate(docs):
        embedding = text_model.encode([doc.page_content])[0].tolist()
        collection.add(
            documents=[doc.page_content],
            embeddings=[embedding],
            ids=[f"{file.name}_{idx}"],
            metadatas=[{"source": file.name, "type": ext}]
        )
    return f"✅ {file.name} indexado con {len(docs)} páginas"

def search_similar_images(image_file, top_k, collection, clip_model, clip_processor, folder):
    with st.spinner("Procesando imagen..."):
        img = Image.open(image_file).convert("RGB")
        emb = get_image_embedding(img, clip_model, clip_processor)
        results = collection.query(query_embeddings=[emb], n_results=top_k)
        st.markdown("### 🖼️ Imágenes similares:")
        for meta in results["metadatas"][0]:
            if meta["type"] == "image":
                st.image(f"{folder}/{meta['source']}", caption=meta["source"])
