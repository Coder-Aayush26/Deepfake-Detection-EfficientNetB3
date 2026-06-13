import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image
import matplotlib.pyplot as plt
import sys

# ── Model Definition ──────────────────────────────────────
class UCFModel(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False)
        in_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.content_enc = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(0.5), nn.Linear(512, 256),
        )
        self.fp_shared = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(0.5),
        )
        self.fp_specific = nn.Linear(512, 256)
        self.fp_common   = nn.Linear(512, 256)
        self.temperature = temperature

        self.common_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(64, 2)
        )
        self.specific_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(64, 2)
        )

    def forward(self, x):
        feat     = self.backbone(x)
        content  = self.content_enc(feat)
        shared   = self.fp_shared(feat)
        specific = F.normalize(self.fp_specific(shared), p=2, dim=1)
        common   = F.normalize(self.fp_common(shared),   p=2, dim=1)
        c_log    = self.common_head(common)     / self.temperature
        s_log    = self.specific_head(specific) / self.temperature
        return c_log, s_log, content, specific, common


# ── Transform ─────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Load Model ────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UCFModel()
model.load_state_dict(torch.load('best_ucf.pth', map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")


# ── Predict Function ──────────────────────────────────────
def predict_image(image_path):
    image  = Image.open(image_path).convert('RGB')
    tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        c_log, _, _, _, _ = model(tensor)
        probs = F.softmax(c_log, dim=1)

    fake_prob = probs[0, 1].item()
    real_prob = probs[0, 0].item()
    label     = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(fake_prob, real_prob)

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    color = 'red' if label == 'FAKE' else 'green'
    plt.title(
        f"Prediction: {label}  (Confidence: {confidence:.1%})\n"
        f"Fake: {fake_prob:.1%}  |  Real: {real_prob:.1%}",
        color=color, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.1%}")
    print(f"Fake       : {fake_prob:.1%}")
    print(f"Real       : {real_prob:.1%}")


# ── Run ───────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])