import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import umap
import streamlit as st

def show_embeddings_pca(collection):
    with st.spinner("Procesando vectores..."):
        results = collection.get(include=["embeddings", "metadatas"])
        embeddings = results["embeddings"]
        metadatas = results["metadatas"]
        labels = [meta["source"] for meta in metadatas]
        tipos = [meta["type"] for meta in metadatas]

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

def show_embeddings_umap(collection):
    with st.spinner("Generando proyección UMAP..."):
        results = collection.get(include=["embeddings", "metadatas"])
        vectors = []
        filenames = []

        for vec, meta in zip(results["embeddings"], results["metadatas"]):
            if meta and meta.get("type") == "image":
                vectors.append(vec)
                filenames.append(meta.get("source", "img"))

        if not vectors:
            st.warning("No hay imágenes vectorizadas para mostrar.")
            return

        coords = umap.UMAP(random_state=42).fit_transform(vectors)

        df = pd.DataFrame({
            "x": [c[0] for c in coords],
            "y": [c[1] for c in coords],
            "Nombre de archivo": filenames,
            "Tipo de documento": ["image"] * len(filenames)
        })

        fig = px.scatter(
            df, x="x", y="y", text="Nombre de archivo", color="Tipo de documento",
            title="🧠 Visualización de Embeddings (UMAP 2D)", height=600
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
