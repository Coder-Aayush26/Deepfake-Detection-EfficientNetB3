from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

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


# ── Load Model ────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UCFModel()
model_path = os.path.join(os.path.dirname(__file__), 'best_ucf_v3.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = val_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            c_log, _, _, _, _ = model(tensor)
            probs = F.softmax(c_log, dim=1)

        fake_prob = round(probs[0, 1].item() * 100, 2)
        real_prob = round(probs[0, 0].item() * 100, 2)
        label = "FAKE" if fake_prob > 50 else "REAL"

        # encode image for preview
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        mime = file.content_type or 'image/jpeg'

        return jsonify({
            'label': label,
            'fake_prob': fake_prob,
            'real_prob': real_prob,
            'image': f'data:{mime};base64,{img_b64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
