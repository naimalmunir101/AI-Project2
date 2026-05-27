import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ==========================================
# DEVICE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# CLASS NAMES & METADATA
# ==========================================
classes = ["Crack", "Normal", "Patchwork", "Pothole"]

class_info = {
    "Crack": {
        "icon": "⚡",
        "color": "#E8739A",
        "bg":    "#FCE4EC",
        "border":"#F48FB1",
        "severity": "High",
        "description": "Linear fractures detected on road surface. Immediate inspection recommended.",
        "action": "Schedule repair within 30 days"
    },
    "Normal": {
        "icon": "🌸",
        "color": "#6DBF9E",
        "bg":    "#E8F5E9",
        "border":"#A5D6A7",
        "severity": "None",
        "description": "Road surface is in good condition. No damage detected.",
        "action": "Continue routine monitoring"
    },
    "Patchwork": {
        "icon": "🔧",
        "color": "#C49BE0",
        "bg":    "#F3E5F5",
        "border":"#CE93D8",
        "severity": "Medium",
        "description": "Previous repair patches identified. Monitor for deterioration.",
        "action": "Monitor quarterly"
    },
    "Pothole": {
        "icon": "🚨",
        "color": "#E07A7A",
        "bg":    "#FFEBEE",
        "border":"#EF9A9A",
        "severity": "Critical",
        "description": "Pothole detected. Poses risk to vehicles and pedestrians.",
        "action": "Immediate repair required"
    }
}

severity_colors = {
    "None":     "#6DBF9E",
    "Medium":   "#C49BE0",
    "High":     "#E8739A",
    "Critical": "#E07A7A"
}

# ==========================================
# IMAGE TRANSFORM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ==========================================
# CNN MODEL
# ==========================================
class RoadCNN(nn.Module):
    def __init__(self):
        super(RoadCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(

    nn.Flatten(),

    nn.Linear(128 * 16 * 16, 256),

    nn.ReLU(),

    nn.Dropout(0.5),

    nn.Linear(256, 4)
)


    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ==========================================
# LOAD MODEL
# ==========================================
model = RoadCNN().to(device)
model.load_state_dict(
    torch.load("best_road_model.pth", map_location=device)
)
model.eval()

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_image(image):
    if image is None:
        return build_placeholder_html()

    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

    label = classes[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100
    info = class_info[label]

    # Confidence bars
    bars_html = ""
    for i, cls in enumerate(classes):
        pct = probabilities[i].item() * 100
        clr = class_info[cls]["color"]
        weight = "800" if cls == label else "500"
        text_color = "#6a5a7a" if cls != label else "#3a2d4a"
        bars_html += f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <span style="font-size:13px;color:{text_color};font-weight:{weight};
                             font-family:'Nunito',sans-serif;">
                    {class_info[cls]['icon']} {cls}
                </span>
                <span style="font-size:13px;color:{clr};font-weight:700;
                             font-family:'Nunito',sans-serif;">{pct:.1f}%</span>
            </div>
            <div style="background:#ede8f5;border-radius:20px;height:8px;overflow:hidden;">
                <div style="width:{pct}%;background:linear-gradient(90deg,{clr}88,{clr});
                            height:100%;border-radius:20px;"></div>
            </div>
        </div>"""

    sev_color = severity_colors[info["severity"]]

    html = f"""
    <div style="
        font-family: 'Nunito', sans-serif;
        background: #fff8fc;
        border: 1.5px solid {info['border']};
        border-radius: 20px;
        padding: 28px;
        color: #3a2d4a;
        max-width: 560px;
        box-shadow: 0 4px 24px {info['color']}22;
    ">
        <!-- Header -->
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:22px;
                    padding-bottom:18px;border-bottom:1.5px dashed {info['border']}88;">
            <div style="
                width:64px;height:64px;border-radius:16px;
                background:{info['bg']};border:1.5px solid {info['border']};
                display:flex;align-items:center;justify-content:center;font-size:30px;
            ">{info['icon']}</div>
            <div>
                <div style="font-size:11px;letter-spacing:2px;color:#c0a8d8;
                            text-transform:uppercase;margin-bottom:4px;font-weight:700;">
                    Detection Result ✨
                </div>
                <div style="font-size:24px;font-weight:800;color:{info['color']};">{label}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
                <div style="font-size:11px;color:#c0a8d8;letter-spacing:2px;
                            text-transform:uppercase;margin-bottom:4px;font-weight:700;">Confidence</div>
                <div style="font-size:26px;font-weight:800;color:#6a3a8a;">{confidence:.1f}%</div>
            </div>
        </div>

        <!-- Severity Badge -->
        <div style="margin-bottom:18px;">
            <span style="
                background:{info['bg']};border:1.5px solid {info['border']};
                color:{sev_color};font-size:12px;letter-spacing:1px;
                text-transform:uppercase;padding:6px 14px;border-radius:20px;font-weight:700;
            ">🎀 Severity: {info['severity']}</span>
        </div>

        <!-- Description Card -->
        <div style="
            background:{info['bg']};border-left:4px solid {info['color']};
            border-radius:0 12px 12px 0;padding:14px 16px;margin-bottom:22px;
        ">
            <div style="font-size:13px;color:#5a4870;line-height:1.7;margin-bottom:8px;">
                {info['description']}
            </div>
            <div style="font-size:12px;color:{info['color']};font-weight:700;">
                → {info['action']}
            </div>
        </div>

        <!-- Confidence Breakdown -->
        <div>
            <div style="font-size:11px;letter-spacing:2px;color:#c0a8d8;
                        text-transform:uppercase;margin-bottom:14px;font-weight:700;">
                💜 Confidence Breakdown
            </div>
            {bars_html}
        </div>

        <!-- Footer -->
        <div style="margin-top:18px;padding-top:14px;border-top:1.5px dashed #ede8f5;
                    display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:10px;color:#c0a8d8;letter-spacing:1px;">🌸 RoadCNN v1.0</span>
            <span style="font-size:10px;color:#c0a8d8;letter-spacing:1px;">Device: {str(device).upper()}</span>
        </div>
    </div>
    """
    return html


def build_placeholder_html():
    return """
    <div style="
        font-family: 'Nunito', sans-serif;
        background: linear-gradient(135deg, #fdf0f8 0%, #f0eeff 100%);
        border: 2px dashed #d8b4e2;
        border-radius: 20px;
        padding: 56px 28px;
        color: #c0a8d8;
        text-align: center;
        max-width: 560px;
    ">
        <div style="font-size:48px;margin-bottom:14px;">🛣️✨</div>
        <div style="font-size:15px;font-weight:800;color:#a889c8;margin-bottom:8px;">
            Upload a Road Image
        </div>
        <div style="font-size:13px;color:#c0a8d8;font-weight:500;">
            The AI will classify the damage for you 🌸
        </div>
    </div>
    """


# ==========================================
# CUSTOM CSS — Pastel / Girly Theme
# ==========================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&family=Quicksand:wght@500;600;700&display=swap');

body, .gradio-container {
    background: linear-gradient(135deg, #fdf4fb 0%, #f3eeff 50%, #fef0f5 100%) !important;
    font-family: 'Nunito', sans-serif !important;
    min-height: 100vh;
}

.gr-panel, .gr-box, .gradio-row, .gradio-column {
    background: transparent !important;
    border: none !important;
}

.gr-image, [data-testid="image"] {
    border: 2px dashed #d8aadc !important;
    border-radius: 16px !important;
    background: #fff8fd !important;
}

.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #d48ac8, #a87ad4) !important;
    color: #ffffff !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 12px 30px !important;
    box-shadow: 0 4px 14px #c49be044 !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}

.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px #c49be066 !important;
}

.gr-button-secondary, button.secondary {
    background: #fff0fa !important;
    color: #c084b8 !important;
    border: 1.5px solid #e8b4d8 !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    border-radius: 30px !important;
}

.gr-html { background: transparent !important; padding: 0 !important; }
footer { display: none !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f9f0ff; }
::-webkit-scrollbar-thumb { background: #d8aadc; border-radius: 10px; }
"""

# ==========================================
# GRADIO UI
# ==========================================
with gr.Blocks(css=custom_css, title="Road Damage Detection 🌸") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:44px 0 16px;">
        <div style="
            display:inline-block;
            font-family:'Nunito',sans-serif;font-size:11px;font-weight:700;
            letter-spacing:3px;color:#c084b8;text-transform:uppercase;
            background:#fff0fa;border:1.5px solid #e8b4d8;
            padding:6px 18px;border-radius:20px;margin-bottom:20px;
        ">🌸 AI-Powered Infrastructure Analysis 🌸</div>

        <h1 style="
            font-family:'Nunito',sans-serif;font-size:2.6rem;font-weight:900;
            background:linear-gradient(135deg,#d472b8,#9b72d4);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;margin:0 0 10px;line-height:1.2;
        ">Road Damage<br>Detection System</h1>

        <p style="
            font-family:'Quicksand',sans-serif;font-size:13px;color:#b09ec8;
            font-weight:600;letter-spacing:2px;text-transform:uppercase;margin:0;
        ">CNN · 4-Class Classification · PyTorch ✨</p>

        <div style="margin-top:18px;display:flex;justify-content:center;gap:8px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#f4a7c3;"></div>
            <div style="width:8px;height:8px;border-radius:50%;background:#c49be0;"></div>
            <div style="width:8px;height:8px;border-radius:50%;background:#89c9e8;"></div>
            <div style="width:8px;height:8px;border-radius:50%;background:#a7ddb8;"></div>
        </div>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="font-family:'Nunito',sans-serif;font-size:11px;font-weight:700;
                        letter-spacing:2px;color:#c084b8;text-transform:uppercase;
                        margin-bottom:10px;">📷 Upload Your Image</div>
            """)
            image_input = gr.Image(type="pil", label="", height=320)
            with gr.Row():
                clear_btn = gr.ClearButton(components=[image_input], value="🗑️ Clear")
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")

        with gr.Column(scale=1):
            gr.HTML("""
            <div style="font-family:'Nunito',sans-serif;font-size:11px;font-weight:700;
                        letter-spacing:2px;color:#c084b8;text-transform:uppercase;
                        margin-bottom:10px;">📋 Analysis Report</div>
            """)
            result_output = gr.HTML(value=build_placeholder_html())

    gr.HTML("""
    <div style="margin-top:36px;padding-top:28px;border-top:2px dashed #e8d0f0;">
        <div style="font-family:'Nunito',sans-serif;font-size:11px;font-weight:700;
                    letter-spacing:3px;color:#c084b8;text-transform:uppercase;
                    text-align:center;margin-bottom:20px;">🎀 Classification Guide 🎀</div>
        <div style="display:flex;gap:14px;justify-content:center;flex-wrap:wrap;">

            <div style="background:#fff8fc;border:1.5px solid #f4b8d0;border-radius:16px;
                        padding:18px 24px;text-align:center;min-width:120px;
                        box-shadow:0 2px 12px #f4b8d022;">
                <div style="font-size:24px;margin-bottom:8px;">⚡</div>
                <div style="font-family:'Nunito',sans-serif;font-weight:800;color:#E8739A;
                            font-size:14px;margin-bottom:4px;">Crack</div>
                <div style="font-size:11px;color:#c09ab8;font-weight:600;">High Severity</div>
            </div>

            <div style="background:#f4fff8;border:1.5px solid #a5d6b0;border-radius:16px;
                        padding:18px 24px;text-align:center;min-width:120px;
                        box-shadow:0 2px 12px #a5d6b022;">
                <div style="font-size:24px;margin-bottom:8px;">🌸</div>
                <div style="font-family:'Nunito',sans-serif;font-weight:800;color:#6DBF9E;
                            font-size:14px;margin-bottom:4px;">Normal</div>
                <div style="font-size:11px;color:#8abcaa;font-weight:600;">No Damage</div>
            </div>

            <div style="background:#fdf4ff;border:1.5px solid #ce93d8;border-radius:16px;
                        padding:18px 24px;text-align:center;min-width:120px;
                        box-shadow:0 2px 12px #ce93d822;">
                <div style="font-size:24px;margin-bottom:8px;">🔧</div>
                <div style="font-family:'Nunito',sans-serif;font-weight:800;color:#C49BE0;
                            font-size:14px;margin-bottom:4px;">Patchwork</div>
                <div style="font-size:11px;color:#a88ec8;font-weight:600;">Medium Severity</div>
            </div>

            <div style="background:#fff5f5;border:1.5px solid #ef9a9a;border-radius:16px;
                        padding:18px 24px;text-align:center;min-width:120px;
                        box-shadow:0 2px 12px #ef9a9a22;">
                <div style="font-size:24px;margin-bottom:8px;">🚨</div>
                <div style="font-family:'Nunito',sans-serif;font-weight:800;color:#E07A7A;
                            font-size:14px;margin-bottom:4px;">Pothole</div>
                <div style="font-size:11px;color:#c09898;font-weight:600;">Critical</div>
            </div>
        </div>
    </div>
    """)

    gr.HTML("""
    <div style="text-align:center;padding:28px 0 10px;
                font-family:'Quicksand',sans-serif;font-size:12px;
                color:#c8b4d8;font-weight:600;letter-spacing:1px;">
        🌸 RoadCNN v1.0 · Built with PyTorch &amp; Gradio · Made with 💜
    </div>
    """)

    analyze_btn.click(fn=predict_image, inputs=image_input, outputs=result_output)
    image_input.change(fn=predict_image, inputs=image_input, outputs=result_output)

# ==========================================
# LAUNCH
# ==========================================
if __name__ == "__main__":
    demo.launch()
