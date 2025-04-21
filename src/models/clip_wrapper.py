import torch
import clip

def load_clip_model(model_name="ViT-B/16"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device