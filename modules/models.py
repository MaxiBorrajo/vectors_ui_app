from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

def load_text_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor
