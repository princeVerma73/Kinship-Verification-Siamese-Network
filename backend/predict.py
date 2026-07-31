from pathlib import Path
import requests
import torch

from .model import SiameseNetwork
from .utils import preprocess_image

PROJECT_DIR = Path(__file__).resolve().parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = PROJECT_DIR / "checkpoints_baseline"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pth"

MODEL_URL = (  #Model size was too long so i uploaded here
    "https://huggingface.co/Prinncever/kinship-verification-siamese/resolve/main/best_model.pth"
)


def download_model():
    if MODEL_PATH.exists():
        print("Model already exists.")
        return

    print("Downloading model from Hugging Face...")

    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    print("Model downloaded successfully.")


download_model()

model = SiameseNetwork().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

print("=" * 60)
print("Checkpoint Type :", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Checkpoint Keys :", checkpoint.keys())

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")

print("Classifier Layer 1 Mean :", model.classifier[0].weight.mean().item())
print("Classifier Layer 2 Mean :", model.classifier[3].weight.mean().item())
print("=" * 60)


def predict(img1_path, img2_path):

    img1 = preprocess_image(img1_path).to(device)
    img2 = preprocess_image(img2_path).to(device)

    with torch.no_grad():

        output = model(img1, img2)

        raw_output = output.item()

        print("Raw output :", raw_output)

        probability = torch.sigmoid(output).item()

        print("Probability :", probability)

    prediction = "Related" if probability >= 0.5 else "Not Related"

    return {
        "prediction": prediction,
        "confidence": round(probability * 100, 2),
    }