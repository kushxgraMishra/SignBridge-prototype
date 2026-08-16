import torch
import numpy as np
import onnxruntime as ort

from model import SignBiLSTM
from dataset import prepare_datasets

device = torch.device("cpu")

print("Loading test data...")
_, _, test_ds, label_encoder, max_len = prepare_datasets()
num_classes = len(label_encoder.classes_)

# --- Load PyTorch model ---
pt_model = SignBiLSTM(input_size=258, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
pt_model.load_state_dict(torch.load("best_model.pth", map_location=device))
pt_model.eval()

# --- Load ONNX model ---
ort_session = ort.InferenceSession("signbridge_model.onnx")

# --- Compare predictions on a few test samples ---
print(f"\nComparing predictions on 5 test samples:\n")
match_count = 0
num_samples = 5

for i in range(num_samples):
    sequence, true_label = test_ds[i]
    true_label = true_label.item()

    # PyTorch prediction
    with torch.no_grad():
        pt_input = sequence.unsqueeze(0)  # add batch dim
        pt_output = pt_model(pt_input)
        pt_pred = pt_output.argmax(1).item()

    # ONNX prediction
    onnx_input = sequence.unsqueeze(0).numpy().astype(np.float32)
    onnx_output = ort_session.run(None, {"landmark_sequence": onnx_input})[0]
    onnx_pred = int(np.argmax(onnx_output, axis=1)[0])

    match = "✓" if pt_pred == onnx_pred else "✗ MISMATCH"
    if pt_pred == onnx_pred:
        match_count += 1

    true_name = label_encoder.classes_[true_label]
    pt_name = label_encoder.classes_[pt_pred]
    onnx_name = label_encoder.classes_[onnx_pred]

    print(f"Sample {i+1}: True={true_name} | PyTorch={pt_name} | ONNX={onnx_name} | {match}")

print(f"\n{match_count}/{num_samples} predictions matched between PyTorch and ONNX")
if match_count == num_samples:
    print("Export verified successfully — ONNX model is ready for deployment.")
else:
    print("Some mismatches found — investigate before using in production.")