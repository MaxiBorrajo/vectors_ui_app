import os
from pathlib import Path
import streamlit as st
from PIL import Image

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from transformers import CLIPProcessor, CLIPModel
import torch

import chromadb
from chromadb.config import Settings

# Config inicial
st.set_page_config(page_title="Buscador Vectorial", layout="wide")
st.title("📚 Buscador Inteligente de Documentos e Imágenes")

# Carpetas y modelos
DOCS_FOLDER = "docs"
os.makedirs(DOCS_FOLDER, exist_ok=True)

text_model = SentenceTransformer("all-MiniLM-L6-v2")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("real_docs")

# Obtener vector de imagen
def get_image_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs[0].tolist()

# Procesar archivo subido
def process_and_store(file):
    ext = file.name.split(".")[-1].lower()
    file_path = os.path.join(DOCS_FOLDER, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())

    if ext == "txt":
        loader = TextLoader(file_path)
        docs = loader.load()
        for idx, doc in enumerate(docs):
            embedding = text_model.encode([doc.page_content])[0].tolist()
            collection.add(
                documents=[doc.page_content],
                embeddings=[embedding],
                ids=[f"{file.name}_{idx}"],
                metadatas=[{"source": file.name, "type": "text"}]
            )
        return f"✅ {file.name} indexado con {len(docs)} páginas"

    elif ext == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for idx, doc in enumerate(docs):
            embedding = text_model.encode([doc.page_content])[0].tolist()
            collection.add(
                documents=[doc.page_content],
                embeddings=[embedding],
                ids=[f"{file.name}_{idx}"],
                metadatas=[{"source": file.name, "type": "pdf"}]
            )
        return f"✅ {file.name} indexado con {len(docs)} páginas"

    elif ext in ["jpg", "jpeg", "png"]:
        image = Image.open(file_path).convert("RGB")
        embedding = get_image_embedding(image)
        collection.add(
            documents=["Imagen subida"],
            embeddings=[embedding],
            ids=[f"{file.name}_img"],
            metadatas=[{"source": file.name, "type": "image"}]
        )
        return f"🖼️ Imagen {file.name} indexada"

    else:
        return f"❌ Formato no soportado: {ext}"

# Subida de archivos
st.subheader("📤 Subí tus archivos (txt, pdf, imagen)")
uploaded_files = st.file_uploader("Archivos válidos: .txt, .pdf, .jpg, .png", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        msg = process_and_store(file)
        st.success(msg)

# ----------------------------
# 🔍 Búsqueda por texto
# ----------------------------
st.subheader("🔍 Hacé una consulta en tus documentos")
query = st.text_input("¿Qué querés buscar?")
top_k = st.slider("Cantidad de resultados", 1, 10, 3)

if st.button("Buscar texto") and query:
    with st.spinner("Buscando..."):
        query_vector = text_model.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        st.markdown("### 📄 Resultados encontrados:")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if meta["type"] in ["text", "pdf"]:
                st.markdown(f"**Fuente:** {meta['source']}")
                st.write(doc)
                st.markdown("---")

# ----------------------------
# 🖼️ Búsqueda por imagen
# ----------------------------
st.subheader("🖼️ Subí una imagen para buscar similares")
image_to_search = st.file_uploader("Imagen para comparar", type=["jpg", "jpeg", "png"])

if image_to_search and st.button("Buscar imágenes similares"):
    with st.spinner("Procesando imagen..."):
        img = Image.open(image_to_search).convert("RGB")
        emb = get_image_embedding(img)
        results = collection.query(query_embeddings=[emb], n_results=top_k)

        st.markdown("### 🖼️ Imágenes similares:")
        for meta in results["metadatas"][0]:
            if meta["type"] == "image":
                st.image(f"{DOCS_FOLDER}/{meta['source']}", caption=meta["source"])

# ----------------------------
# 📊 Visualización de vectores
# ----------------------------
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px

st.subheader("📊 Visualización 2D de vectores")

if st.button("Mostrar embeddings en 2D"):
    with st.spinner("Procesando vectores..."):
        results = collection.get(include=["embeddings", "metadatas"])
        embeddings = results["embeddings"]
        metadatas = results["metadatas"]
        labels = [meta["source"] for meta in metadatas]
        tipos = [meta["type"] for meta in metadatas]

        # Reducción de dimensión
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        df = pd.DataFrame(reduced, columns=["x", "y"])
        df["label"] = labels
        df["tipo"] = tipos

        fig = px.scatter(
            df, x="x", y="y", color="tipo", text="label",
            title="📊 Visualización de Embeddings (PCA 2D)",
            labels={"tipo": "Tipo de documento"}
        )
        st.plotly_chart(fig, use_container_width=True)

import umap
import plotly.express as px

st.subheader("🗺️ Visualización de Embeddings (UMAP)")

# if st.button("Visualizar imágenes con UMAP"):
#     with st.spinner("Generando proyección UMAP..."):
#         # Obtener todos los embeddings e info de imágenes
#         results = collection.get(include=["embeddings", "metadatas"])
#         vectors = []
#         filenames = []

#         if results["embeddings"] and results["metadatas"]:
#             for vec, meta in zip(results["embeddings"], results["metadatas"]):
#                 if meta and meta.get("type") == "image":
#                     vectors.append(vec)
#                     filenames.append(meta.get("source", "img"))

#         if vectors:
#             reducer = umap.UMAP(random_state=42)
#             coords = reducer.fit_transform(vectors)

#             import pandas as pd
#             import plotly.express as px

#             df = pd.DataFrame({
#                 "x": [c[0] for c in coords],
#                 "y": [c[1] for c in coords],
#                 "Nombre de archivo": filenames,
#                 "Tipo de documento": ["image"] * len(filenames)
#             })

#             fig = px.scatter(
#                 df, x="x", y="y", text="Nombre de archivo", color="Tipo de documento",
#                 title="🧠 Visualización de Embeddings (UMAP 2D)",
#                 height=600
#             )
#             fig.update_traces(textposition='top center')
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.warning("No hay imágenes vectorizadas para mostrar.")
