import torch
import json
from model import SignBiLSTM
from dataset import prepare_datasets

device = torch.device("cpu")

# --- Load label encoder (needed to map model output indices back to sign words) ---
print("Loading dataset metadata (for label mapping and sequence length)...")
_, _, _, label_encoder, max_len = prepare_datasets()
num_classes = len(label_encoder.classes_)

print(f"Classes: {num_classes}, Sequence length: {max_len}")

# --- Load trained model ---
model = SignBiLSTM(input_size=258, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("signbridge_final.pth", map_location=device))
model.eval()

# --- Save label mapping so your FastAPI backend can decode predictions ---
label_map = {int(i): str(cls) for i, cls in enumerate(label_encoder.classes_)}
with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)
print("Saved label_map.json")

# --- Dummy input matching your model's expected shape: (batch, seq_len, features) ---
dummy_input = torch.randn(1, max_len, 258)

# --- Export to ONNX ---
onnx_path = "signbridge_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["landmark_sequence"],
    output_names=["class_logits"],
    dynamic_axes={
        "landmark_sequence": {0: "batch_size"},   # allow variable batch size at inference
        "class_logits": {0: "batch_size"}
    },
    opset_version=13
)

print(f"Exported model to {onnx_path}")
print(f"Input shape expected: (batch_size, {max_len}, 258)")
print(f"Output shape: (batch_size, {num_classes})")