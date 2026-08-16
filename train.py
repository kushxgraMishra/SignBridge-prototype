import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import prepare_datasets
from model import SignBiLSTM
from augmentation import get_weighted_sampler  # NEW

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Load data ---
train_ds, val_ds, test_ds, label_encoder, max_len = prepare_datasets()

# NEW: build a sampler that oversamples low-count classes so training
# doesn't just learn to guess your most common signs.
train_sampler = get_weighted_sampler(train_ds.labels.numpy())

train_loader = DataLoader(train_ds, batch_size=16, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# --- Model, loss, optimizer ---
model = SignBiLSTM(input_size=258, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 60
best_val_loss = float("inf")
best_val_acc = 0.0
epochs_without_improvement = 0
early_stop_patience = 5  # stop if val_loss doesn't improve for this many epochs

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    total_train_loss = 0
    correct_train = 0

    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        correct_train += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct_train / len(train_ds)
    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    correct_val = 0

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            correct_val += (outputs.argmax(1) == labels).sum().item()

    val_acc = correct_val / len(val_ds)
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")

    # --- Save best model by val_loss (used for early stopping) ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  -> Saved new best model by val_loss (val_loss={avg_val_loss:.4f})")
    else:
        epochs_without_improvement += 1

    # --- Save best model by val_acc (separate checkpoint) ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_by_acc.pth")
        print(f"  -> Saved new best model by val_acc (val_acc={val_acc:.2%})")

    # --- Early stopping ---
    if epochs_without_improvement >= early_stop_patience:
        print(f"\nNo val_loss improvement for {early_stop_patience} epochs. Stopping early at epoch {epoch+1}.")
        break

print(f"\nTraining complete.")
print(f"Best model by val_loss saved to best_model.pth (val_loss={best_val_loss:.4f})")
print(f"Best model by val_acc saved to best_model_by_acc.pth (val_acc={best_val_acc:.2%})")