import torch

from model import SiameseNetwork
from utils import preprocess_image

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = SiameseNetwork().to(device)

# Load checkpoint
checkpoint = torch.load(
    "checkpoints_baseline/best_model.pth",
    map_location=device
)

print("=" * 60)
print("Checkpoint Type :", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Checkpoint Keys :", checkpoint.keys())

# Load weights
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

        print("Raw output :", output.item())
        probability = torch.sigmoid(output).item()
        print("Probability :", probability)

    print("\n" + "=" * 60)
    print("Image 1 :", img1_path)
    print("Image 2 :", img2_path)
    print("Raw Output :", raw_output)
    print("Probability :", probability)
    print("=" * 60)

    prediction = (
        "Related"
        if probability >= 0.5
        else "Not Related"
    )

    return {
        "prediction": prediction,
        "confidence": round(probability * 100, 2)
    }