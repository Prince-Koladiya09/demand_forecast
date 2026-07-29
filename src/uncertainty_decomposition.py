"""
uncertainty_decomposition.py
------------------------------
Decomposes total predictive uncertainty into:
    - Epistemic (model) uncertainty: uncertainty due to limited training
      data / model not being sure of its own parameters. Reducible with
      more data or a better model.
    - Aleatoric (data) uncertainty: uncertainty inherent to the demand
      process itself (genuine randomness in customer behavior). NOT
      reducible, no matter how much data you add.

Why this matters: a wide prediction interval could mean two very different
things -- "the model hasn't seen enough examples like this" (epistemic,
fixable) or "this SKU's demand is just inherently noisy" (aleatoric, not
fixable). Business teams make different decisions depending on which one
it is, so collapsing both into a single interval hides useful information.

Method:
- Epistemic uncertainty is estimated via Monte Carlo Dropout (Gal & Ghahramani,
  2016): dropout is normally OFF at inference (model.eval()). MC Dropout
  instead keeps dropout ON and runs many stochastic forward passes on the
  SAME input. The spread across those passes approximates the model's
  uncertainty about its own parameters -- this is a well-established,
  cheap approximation to a Bayesian neural network, not a full Bayesian
  treatment.
- Aleatoric uncertainty is approximated from the model's own predicted
  quantile spread (P90 - P10) at each pass, averaged across passes. Since
  quantile regression is explicitly trained to model the target
  distribution's spread, this spread is a reasonable (if approximate)
  proxy for the data's inherent noise -- NOT a separate, independently
  validated aleatoric estimator. This is a pragmatic choice appropriate
  for this project's scope, not a claim of statistical exactness.

This is an approximation suitable for demonstrating the CONCEPT of
uncertainty decomposition, not a fully rigorous Bayesian deep learning
implementation (that would require a proper variational or ensemble
approach, which is out of scope here -- see docs/SRS.md's design
philosophy of correctness over unnecessary complexity).
"""

import numpy as np
import torch

from src import dl_model


def enable_mc_dropout(model):
    """
    Standard MC Dropout trick: put the model in train() mode (which keeps
    Dropout layers active) WITHOUT actually training -- no gradients are
    computed and no optimizer step happens. This model has no BatchNorm
    layers, so train() mode only affects Dropout, which is exactly what we
    want here.
    """
    model.train()
    return model


def mc_dropout_predict(model, X, n_passes: int = 30, device: str = "cpu"):
    """
    Runs n_passes stochastic forward passes (dropout active) on the same
    input X and returns:
        mean_preds:       (n_samples, n_quantiles, output_window) -- average prediction
        epistemic_std:    (n_samples, output_window) -- std across passes of the P50 prediction
                           (model uncertainty: how much does the prediction
                           itself change pass-to-pass due to dropout)
        aleatoric_spread: (n_samples, output_window) -- mean across passes of (P90 - P10)
                           (data uncertainty: how wide is the predicted
                           distribution itself, on average)
    """
    model = enable_mc_dropout(model)
    model.to(device)

    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)

    all_preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds = model(X_tensor)  # (n, 3, output_window)
            all_preds.append(preds.cpu().numpy())

    stacked = np.stack(all_preds, axis=0)  # (n_passes, n, 3, output_window)

    mean_preds = stacked.mean(axis=0)                      # (n, 3, output_window)
    p50_across_passes = stacked[:, :, 1, :]                 # (n_passes, n, output_window)
    epistemic_std = p50_across_passes.std(axis=0)           # (n, output_window)

    spread_per_pass = stacked[:, :, 2, :] - stacked[:, :, 0, :]  # P90 - P10 per pass
    spread_per_pass = np.maximum(spread_per_pass, 0)  # a "spread" is never negative; guards against
                                                          # occasional quantile-crossing in an undertrained model
    aleatoric_spread = spread_per_pass.mean(axis=0)          # (n, output_window)

    model.eval()  # restore normal inference mode for any subsequent calls
    return mean_preds, epistemic_std, aleatoric_spread


def summarize_by_horizon(epistemic_std, aleatoric_spread):
    """
    Average both uncertainty components across all test samples, per
    forecast horizon step (t+1, t+2, ..., t+output_window). This is what
    reveals patterns like "epistemic uncertainty is highest at t+1 (model
    isn't sure about the immediate next value) while aleatoric uncertainty
    grows toward t+7 (demand naturally gets noisier further out)".
    """
    return {
        "epistemic_by_horizon": epistemic_std.mean(axis=0),   # (output_window,)
        "aleatoric_by_horizon": aleatoric_spread.mean(axis=0),  # (output_window,)
    }


if __name__ == "__main__":
    # Minimal smoke test with random data
    batch_size, input_window, output_window = 16, 28, 7
    dummy_X = np.random.randn(batch_size, input_window, 1).astype(np.float32)

    model = dl_model.QuantileLSTM(output_window=output_window, dropout=0.3)

    mean_preds, epistemic_std, aleatoric_spread = mc_dropout_predict(model, dummy_X, n_passes=10)
    print("mean_preds shape:", mean_preds.shape)
    print("epistemic_std shape:", epistemic_std.shape)
    print("aleatoric_spread shape:", aleatoric_spread.shape)

    summary = summarize_by_horizon(epistemic_std, aleatoric_spread)
    print("Epistemic uncertainty by horizon:", summary["epistemic_by_horizon"])
    print("Aleatoric uncertainty by horizon:", summary["aleatoric_by_horizon"])
