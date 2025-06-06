# 📚 Buscador Inteligente de Documentos e Imágenes

Este proyecto es una app de Streamlit que permite **indexar y buscar documentos de texto, PDFs e imágenes usando embeddings vectoriales**. Utiliza modelos preentrenados de lenguaje e imagen (CLIP y Sentence Transformers), y almacena los vectores en una base de datos local usando **ChromaDB**.

---

## 🚀 Funcionalidades

- 📤 Subida e indexación de:
  - Archivos `.txt`
  - PDFs (`.pdf`)
  - Imágenes (`.jpg`, `.jpeg`, `.png`)
- 🔍 Búsqueda semántica por texto (encuentra documentos relacionados con una consulta).
- 🖼️ Búsqueda de imágenes similares mediante embeddings visuales (CLIP).

---

## 🧰 Tecnologías usadas

- [Streamlit](https://streamlit.io/) — Interfaz web
- [Sentence Transformers](https://www.sbert.net/) — Embeddings de texto
- [CLIP (OpenAI)](https://huggingface.co/openai/clip-vit-base-patch32) — Embeddings de imágenes
- [ChromaDB](https://docs.trychroma.com/) — Base vectorial local
- [LangChain Loaders](https://python.langchain.com/) — Carga de archivos

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/buscador-vectorial.git
cd buscador-vectorial
```

### 2. Crear un entorno virtual (opcional pero recomendado)

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota**: si no tenés `requirements.txt`, podés generarlo desde el entorno actual:

```bash
pip freeze > requirements.txt
```

---

## ▶️ Cómo ejecutar

```bash
streamlit run app.py
```

> El archivo principal se llama `app.py`. Si lo nombraste distinto, ajustá el comando.

---

## 📂 Estructura esperada

```bash
buscador-vectorial/
├── app.py               # Archivo principal
├── docs/                # Carpeta donde se guardan los archivos subidos
├── requirements.txt     # Lista de dependencias
└── README.md
```

---

## 📦 Recomendaciones

- Usá imágenes pequeñas o comprimidas para pruebas rápidas.
- Los documentos se guardan en la carpeta `docs/`.
- Si reiniciás la app, el contenido indexado permanece mientras no borres `docs/` ni el almacenamiento de ChromaDB.

---

## 🧪 Ejemplo de uso

1. Subí un archivo `resumen_redes.txt`.
2. Ingresá: `"protocolos de capa de red"` como búsqueda.
3. La app mostrará fragmentos de texto relacionados.
4. Subí una imagen y luego otra parecida → buscará las visualmente similares.

---

## 📋 Notas

- Los embeddings se generan **en tiempo real**, por lo tanto, puede haber una pequeña demora en búsquedas o al subir documentos grandes.
- La base es local, no requiere servidores externos ni conexión a internet para buscar.

---

## 📫 Contacto

Desarrollado por **CUKI** 🧠  
Si tenés preguntas o querés mejorar este proyecto, ¡hacé un fork o escribime!
