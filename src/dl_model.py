"""
dl_model.py
-----------
Implements FR-3: Deep Learning Model.

A single LSTM sequence model with THREE output heads, one per quantile
(P10, P50, P90). Trained with pinball (quantile) loss instead of MSE, so
the model learns a distribution instead of a single mean estimate.

Design choice: quantile regression was chosen over MC Dropout / a full
Bayesian NN because (a) it is simpler to implement correctly, (b) it
trains faster, and (c) its output plugs directly into the newsvendor
decision formula (FR-6) without needing an assumed distribution shape.
"""

import torch
import torch.nn as nn

QUANTILES = [0.1, 0.5, 0.9]


class QuantileLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_window=7,
                 quantiles=QUANTILES, dropout=0.2):
        """
        dropout: applied to the LSTM's final hidden state before the quantile
        heads. Kept even at inference time when running Monte Carlo Dropout
        (see src/uncertainty_decomposition.py) -- this is what lets a single
        deterministic-looking model produce a spread of predictions used to
        estimate epistemic (model) uncertainty. Standard training/prediction
        (dl_model.predict()) uses model.eval(), which disables dropout as
        usual -- only the MC Dropout module deliberately re-enables it.
        """
        super().__init__()
        self.quantiles = quantiles
        self.output_window = output_window
        self.dropout_p = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # One linear head per quantile, each predicting the full output window
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, output_window) for _ in quantiles
        ])

    def forward(self, x):
        """
        x shape: (batch, input_window, input_size)
        returns: (batch, num_quantiles, output_window)
        """
        _, (h_n, _) = self.lstm(x)
        last_hidden = self.dropout(h_n[-1])  # (batch, hidden_size)
        outputs = [head(last_hidden) for head in self.heads]  # each (batch, output_window)
        return torch.stack(outputs, dim=1)  # (batch, num_quantiles, output_window)


def pinball_loss(preds, target, quantiles=QUANTILES):
    """
    Pinball (quantile) loss, computed per quantile and averaged.

    preds:  (batch, num_quantiles, output_window)
    target: (batch, output_window)

    For quantile q:
        loss = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    This penalizes under- and over-prediction asymmetrically depending
    on which quantile is being predicted.
    """
    target = target.unsqueeze(1)  # (batch, 1, output_window) for broadcasting
    losses = []
    for i, q in enumerate(quantiles):
        error = target[:, 0, :] - preds[:, i, :]
        loss_q = torch.max((q - 1) * error, q * error)
        losses.append(loss_q.mean())
    return torch.stack(losses).mean()


def train_model(model, train_loader, val_loader=None, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = pinball_loss(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        msg = f"Epoch {epoch + 1}/{epochs} - train pinball loss: {train_loss:.4f}"

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    preds = model(X_val)
                    val_loss += pinball_loss(preds, y_val).item() * X_val.size(0)
            val_loss /= len(val_loader.dataset)
            msg += f" - val pinball loss: {val_loss:.4f}"

        print(msg)

    return model


def predict(model, X, device="cpu"):
    """
    X: shape (n_samples, input_window, n_channels) -- as produced by
    src/preprocessing.py's make_windows(). Channel 0 is the target,
    any remaining channels are extra features (e.g. price, promo).

    Returns quantile predictions as a numpy array of shape
    (num_samples, num_quantiles, output_window).
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(-1)  # backward-compat: add channel dim if a plain 2D series is passed
        preds = model(X_tensor)
    return preds.cpu().numpy()


if __name__ == "__main__":
    # Minimal smoke test with random tensors
    batch_size, input_window, output_window = 8, 28, 7
    dummy_X = torch.randn(batch_size, input_window, 1)
    dummy_y = torch.randn(batch_size, output_window)

    model = QuantileLSTM(output_window=output_window)
    preds = model(dummy_X)
    print("Prediction shape:", preds.shape)  # expected: (8, 3, 7)

    loss = pinball_loss(preds, dummy_y)
    print("Pinball loss:", loss.item())