import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

from dataset import prepare_datasets
from model import SignBiLSTM

def evaluate_checkpoint(checkpoint_path, model, test_loader, criterion, device,
                         num_classes, label_encoder, test_ds):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    total_loss = 0
    correct = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

            for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    test_acc = correct / len(test_ds)
    avg_test_loss = total_loss / len(test_loader)

    print(f"\n=== {checkpoint_path} ===")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2%}  ({correct}/{len(test_ds)})")

    per_class_acc = []
    for class_idx in class_total:
        acc = class_correct[class_idx] / class_total[class_idx]
        class_name = label_encoder.classes_[class_idx]
        per_class_acc.append((class_name, acc, class_correct[class_idx], class_total[class_idx]))
    per_class_acc.sort(key=lambda x: x[1])

    print(f"\nWorst-performing classes:")
    for name, acc, correct_n, total_n in per_class_acc[:10]:
        print(f"  {name:20s} {acc:6.1%}  ({correct_n}/{total_n})")

    missing = set(range(num_classes)) - set(class_total.keys())
    if missing:
        missing_names = [label_encoder.classes_[c] for c in missing]
        print(f"\nNote: {len(missing)} classes had no test samples: {missing_names}")

    return test_acc, avg_test_loss


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Load data (same split as training, since random_state=42 is fixed) ---
train_ds, val_ds, test_ds, label_encoder, max_len = prepare_datasets()
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Test set size: {len(test_ds)}")

# --- Build model once, reload weights per checkpoint ---
model = SignBiLSTM(input_size=258, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()

results = {}
for ckpt in ["best_model.pth", "best_model_by_acc.pth"]:
    if os.path.exists(ckpt):
        acc, loss = evaluate_checkpoint(ckpt, model, test_loader, criterion, device,
                                         num_classes, label_encoder, test_ds)
        results[ckpt] = (acc, loss)
    else:
        print(f"\n{ckpt} not found, skipping.")

if len(results) == 2:
    print(f"\n=== COMPARISON ===")
    for ckpt, (acc, loss) in results.items():
        print(f"  {ckpt:25s} Test Acc: {acc:.2%}  Test Loss: {loss:.4f}")
    best_ckpt = max(results, key=lambda k: results[k][0])
    print(f"\nBest on test set: {best_ckpt} ({results[best_ckpt][0]:.2%})")