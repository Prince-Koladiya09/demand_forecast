
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = [0.1, 0.5, 0.9]


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QuantileLSTM(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=1,
        output_window=7,
        quantiles=QUANTILES,
        dropout=0.2
    ):
        super().__init__()

        if sorted(quantiles) != list(quantiles):
            raise ValueError("Quantiles must be sorted")

        self.quantiles = quantiles
        self.output_window = output_window
        self.dropout_p = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        # First head predicts the lowest quantile.
        self.base_head = nn.Linear(
            hidden_size, output_window
        )

        # Positive increments guarantee non-crossing.
        self.increment_heads = nn.ModuleList([
            nn.Linear(hidden_size, output_window)
            for _ in range(len(quantiles) - 1)
        ])

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        hidden = self.dropout(hidden[-1])

        base = self.base_head(hidden)

        outputs = [base]
        current = base

        for head in self.increment_heads:

            increment = F.softplus(head(hidden))

            current = current + increment

            outputs.append(current)

        return torch.stack(outputs, dim=1)


def pinball_loss(
    preds,
    target,
    quantiles=QUANTILES
):
    target = target.unsqueeze(1)

    losses = []

    for i, q in enumerate(quantiles):

        error = target[:, 0, :] - preds[:, i, :]

        loss = torch.maximum(
            q * error,
            (q - 1) * error
        )

        losses.append(loss.mean())

    return torch.stack(losses).mean()


def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=20,
    lr=1e-3,
    device="cpu",
    patience=5,
    seed=42,
    min_delta=1e-5
):
    set_global_seed(seed)

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )

    best_loss = float("inf")
    best_state = None
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):

        model.train()

        train_total = 0.0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            predictions = model(xb)

            loss = pinball_loss(
                predictions,
                yb,
                model.quantiles
            )

            loss.backward()

            optimizer.step()

            train_total += (
                loss.item() * xb.size(0)
            )

        train_loss = (
            train_total / len(train_loader.dataset)
        )

        history["train_loss"].append(train_loss)

        val_loss = train_loss

        if val_loader is not None:

            model.eval()

            val_total = 0.0

            with torch.no_grad():

                for xb, yb in val_loader:

                    xb = xb.to(device)
                    yb = yb.to(device)

                    predictions = model(xb)

                    loss = pinball_loss(
                        predictions,
                        yb,
                        model.quantiles
                    )

                    val_total += (
                        loss.item() * xb.size(0)
                    )

            val_loss = (
                val_total / len(val_loader.dataset)
            )

            history["val_loss"].append(val_loss)

            if val_loss < best_loss - min_delta:

                best_loss = val_loss

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                wait = 0

            else:
                wait += 1

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Train={train_loss:.5f} "
            f"Validation={val_loss:.5f}"
        )

        if val_loader is not None and wait >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.training_history = history
    model.best_val_loss = best_loss

    return model


def predict(model, X, device="cpu"):

    model.eval()
    model.to(device)

    with torch.no_grad():

        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=device
        )

        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(-1)

        predictions = model(X_tensor)

    return predictions.cpu().numpy()
