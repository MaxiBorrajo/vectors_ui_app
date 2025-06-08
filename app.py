import os
from pathlib import Path
from PIL import Image
import streamlit as st

from modules.models import load_text_model, load_clip_model
from modules.vector_store import init_vector_store
from modules.embed import process_and_store_file, search_similar_images
from modules.visualization import show_embeddings_pca, show_embeddings_umap

# Configuración inicial
st.set_page_config(page_title="Buscador Vectorial", layout="wide")
st.title("📚 Buscador Inteligente de Documentos e Imágenes")

# Inicialización de modelos y base
DOCS_FOLDER = "docs"
os.makedirs(DOCS_FOLDER, exist_ok=True)

text_model = load_text_model()
clip_model, clip_processor = load_clip_model()
collection = init_vector_store()

# Subida de archivos
st.subheader("📤 Subí tus archivos (imagen)")
uploaded_files = st.file_uploader("Archivos válidos: .jpg, .png", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        msg = process_and_store_file(file, DOCS_FOLDER, collection, text_model, clip_model, clip_processor)
        st.success(msg)


# Búsqueda por imagen
st.subheader("🖼️ Subí una imagen para buscar similares")
image_to_search = st.file_uploader("Imagen para comparar", type=["jpg", "jpeg", "png"])
top_k = st.slider("Cantidad de resultados", 1, 10, 3)
if image_to_search and st.button("Buscar imágenes similares"):
    search_similar_images(image_to_search, top_k, collection, clip_model, clip_processor, DOCS_FOLDER)
# Visualización PCA
st.subheader("📊 Visualización 2D de vectores")
if st.button("Mostrar embeddings en 2D"):
    show_embeddings_pca(collection)

# Visualización UMAP (opcional)
st.subheader("🗺️ Visualización de Embeddings (UMAP)")
# if st.button("Visualizar imágenes con UMAP"):
#     show_embeddings_umap(collection)
