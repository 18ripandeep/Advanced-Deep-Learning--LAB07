# Latent Spaces

A Variational Autoencoder (VAE) written from scratch on 8 by 8 digits (`sklearn.datasets.load_digits`), compared against a plain autoencoder with an identical bottleneck, swept across KL weights, probed via latent interpolation and class conditioning, and evaluated across multiple generative quality metrics.

## Setup

    python -m venv .venv
    source .venv/bin/activate
    # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Run

    python vae_checks.py      # piece 1: KL, gradient and ELBO checks
    python ae_vs_vae.py       # piece 2: same bottleneck comparison (AE vs VAE)
    python beta_sweep.py      # piece 3: KL weight sweep and posterior collapse
    python probe_latent.py    # piece 4: latent interpolation and conditional VAE
    python metrics_study.py   # piece 5: what the metrics capture and sample-size sensitivity

## Hardware

- **Processor**: Laptop CPU
- **Runtime**: ~3 to 5 minutes total for all training runs across 3 seeds

## Conventions

- **Dataset Split**: 1,438 training images and 359 test images (8x8 pixels scaled to `[0, 1]`, flattened to 64 values) via `load_task(seed=0)`.
- **Reconstruction Term**: Binary cross-entropy with logits summed over the 64 pixels and averaged over the batch.
- **KL Divergence Term**: Closed-form KL to a standard normal N(0, I), summed over the 8 latent dimensions and averaged over the batch.
- **Evaluation Consistency**: All Fréchet distances, precision, and recall scores are computed using the exact same `FeatureNet` (`seed=0`) and 359 generated samples unless testing sample-size sensitivity in Piece 5.

## Prediction (written before running the sweep)

Before running the beta sweep (`0.1, 0.5, 1.0, 2.0, 4.0`), I predict:
1. Increasing beta will increase reconstruction loss while decreasing the KL divergence term and shrinking the number of active latent units (variance > 0.01).
2. Complete posterior collapse (0 active units, where the decoder ignores z and outputs an average blurry digit) will occur at beta = 2.0 or beta = 4.0.
3. beta = 0.5 or beta = 1.0 will produce the lowest Fréchet distance and sharpest prior samples, balancing recognizable digit structure with smooth coverage of N(0, I).

## Results

### Piece 1: Verification Checks
- **Closed-form KL vs. Monte Carlo KL**: Closed-form = `6.5805`, Monte Carlo = `6.5815` (agree within sampling noise).
- **Encoder Mean Gradient Norm**: With reparameterization trick = `4.6129` (> 0); with detached latent sample = `0.0000`.
- **Negative ELBO**: Verified that `recon_loss + 1.0 * kl_loss` evaluates successfully to `44.5212`.

### Piece 2: Plain Autoencoder vs. VAE (beta = 1.0, 200 epochs)

| Model | Seed | Test Recon MSE / Term | Fréchet Distance | Precision | Recall | Nearest Train Dist |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Plain AE | 0 | 0.0150 | 84.10 | 0.354 | 0.786 | 2.129 |
| Plain AE | 1 | 0.0141 | 69.26 | 0.485 | 0.791 | 2.081 |
| Plain AE | 2 | 0.0146 | 71.35 | 0.370 | 0.702 | 2.107 |
| VAE (beta=1) | 0 | 0.0264 | 14.30 | 0.939 | 0.373 | 1.116 |
| VAE (beta=1) | 1 | 0.0263 | 12.02 | 0.933 | 0.521 | 1.154 |
| VAE (beta=1) | 2 | 0.0269 | 13.96 | 0.900 | 0.493 | 1.148 |

- **Code Spread**: The plain AE encodes data into unregularized regions with uneven variances across dimensions, so random draws from N(0, I) land in empty gaps and decode to noise. The VAE regularizes its posterior toward N(0, I), keeping code variances balanced so prior samples decode into coherent digits.

### Piece 3: KL Weight Sweep

| Beta | Seed | Recon Term | KL Term | Active Units | Fréchet Distance |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.1** | 0 / 1 / 2 | 17.08 / 17.16 / 17.17 | 14.30 / 14.19 / 14.24 | 8 / 8 / 8 | 11.72 / 10.79 / 11.14 |
| **0.5** | 0 / 1 / 2 | 19.12 / 19.20 / 19.17 | 6.06 / 6.01 / 6.08 | 7 / 7 / 7| 8.05 / 9.16 / 11.17 |
| **1.0** | 0 / 1 / 2 | 20.91 / 20.95 / 20.99 | 3.33 / 3.49 / 3.43 | 5 / 5 / 5 | 12.07 / 14.09 / 13.09 |
| **2.0** | 0 / 1 / 2 | 24.37 / 24.55 / 24.03 | 1.26 / 1.16 / 1.19 | 3 / 3 / 3 | 44.53 / 45.18 / 43.53 |
| **4.0** | 0 / 1 / 2 | 27.25 / 27.24 / 27.24| 0.00 / 0.00 / 0.00| 0 / 0 / 0 | 226.74 / 217.51 / 217.71 |

- **Collapse Analysis**: Complete posterior collapse occurs at beta = 4.0, where active units drop to `0`, KL divergence vanishes to `0.00`, and Fréchet distance spikes above `217` as the model produces identical blurry outputs (saved in `collapsed_vae.png`).

### Piece 4: Probing the Latent Space & Conditioning
- **Interpolation (40 pairs, 10 steps across seeds 0, 1, 2)**[cite: 19]:
  - **VAE Latent Path**: Mean confidence = `0.9070`, Mean class changes = `1.77`
  - **AE Latent Path**: Mean confidence = `0.9152`, Mean class changes = `1.77`
  - **Pixel Cross-Fade**: Mean confidence = `0.9365`, Mean class changes = `1.32`
- **Conditional VAE (CVAE)**: Across 2,000 conditional test samples (200 per digit), `FeatureNet` achieves a conditional accuracy share of `1.000` (100% correct requested digit generation), far outperforming unconditional draws.

### Piece 5: What the Metrics Capture

| Image Set (359 images) | Fréchet Distance | Precision | Recall | Nearest Train Dist |
| :--- | :---: | :---: | :---: | :---: |
| Real Test Set | `-5.46e-12`| `1.000` | `1.000` | `1.068`|
| Copier (Train + sigma=0.02) | `2.360` | `0.919` | `0.922` | `0.135` |
| Mean + Noise (sigma=0.1) | `138.186` | `0.939`| `0.000`| `1.616`|

- **Fréchet Distance vs. Subset Size (10 subsets each)**:
  - N = 30: Mean FD = `21.18` (std `6.89`)
  - N = 60: Mean FD = `11.32` (std `5.35`)
  - N = 120: Mean FD = `5.77` (std `1.38`)
  - N = 359: Mean FD = `2.81` (std `0.88`)
  - N = 1000: Mean FD = `1.79` (std `0.22`)
- **Takeaway**: Fréchet distance and precision fail to catch a copier (which scores `2.360` FD and `0.919` precision), requiring `nearest_train` distance (`0.135`) to detect memorization[cite: 18]. Conversely, a noisy mean image fails on recall (`0.000` — zero diversity) and Fréchet distance (`138.186`) despite high precision (`0.939`). Classification accuracy cannot evaluate unconditional generation because generated samples have no ground-truth target labels.

## Figures

- `starter_check.png`: Plain autoencoder samples decoded from N(0, I).
- `collapsed_vae.png`: Collapsed VAE samples at beta = 4.0 showing identical outputs.

## Conclusion

Training on the ELBO with the reparameterization trick forces the VAE's latent space to align with the N(0, I) prior, turning a compression bottleneck into a working generative model at the cost of slightly higher reconstruction error than a plain autoencoder. Evaluating generative models requires combining distributional distance (Fréchet), fidelity (precision), diversity (recall), and memorization checks (nearest training distance), alongside strictly fixed sample sizes and feature extractors.

## Known limitations

All findings are bounded by the low-resolution 8x8 digits dataset, a single-hidden-layer MLP architecture (64 units), a self-trained lightweight `FeatureNet` rather than a pretrained deep perceptual network, a fixed 200-epoch budget, and three random seeds.
