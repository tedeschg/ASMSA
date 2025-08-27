import math
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Utils
# ==============================
def _to_numpy(x):
    if hasattr(x, "numpy"):
        x = x.numpy()
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.reshape(-1)
    return x

def _safe_corrcoef(x, y):
    sx = np.std(x); sy = np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def circular_diff_rad(y_true_rad, y_pred_rad):
    diff = np.abs(y_true_rad - y_pred_rad)
    return np.minimum(diff, 2 * np.pi - diff)

def huber_numpy(error, delta=1.0):
    abs_e = np.abs(error)
    quadratic = np.minimum(abs_e, delta)
    linear = abs_e - quadratic
    return 0.5 * quadratic**2 + delta * linear

# -------------------------------
# Angles from sin/cos blocks
# -------------------------------
def _angles_from_sincos_blocks(vec_ang, n_phi, n_psi):
    """
    Converts vector [phi_sin (n_phi), phi_cos (n_phi),
                     psi_sin (n_psi), psi_cos (n_psi)]
    into an array of angles (radians).
    """
    off = 0
    phi_sin = vec_ang[off:off+n_phi];  off += n_phi
    phi_cos = vec_ang[off:off+n_phi];  off += n_phi
    psi_sin = vec_ang[off:off+n_psi];  off += n_psi
    psi_cos = vec_ang[off:off+n_psi];  off += n_psi

    phi = np.arctan2(phi_sin, phi_cos)
    psi = np.arctan2(psi_sin, psi_cos)

    labels = [f"phi{i}" for i in range(n_phi)] + [f"psi{i}" for i in range(n_psi)]
    return np.concatenate([phi, psi]), labels

# ==============================
# Main analysis
# ==============================
def analyze_reconstruction_with_sincos_blocks(
    orig, recon, nD, n_phi, n_psi,
    feature_names_dist=None,
    deltaD=0.1, deltaA=0.5,
    title_prefix=""
):
    orig = _to_numpy(orig); recon = _to_numpy(recon)
    d_true = orig[:nD]; d_pred = recon[:nD]
    a_true = orig[nD:]; a_pred = recon[nD:]
    nA = len(a_true)

    # --- DISTANCES ---
    if nD > 0:
        d_err = d_true - d_pred
        d_abs = np.abs(d_err)
        d_mse = float(np.mean(d_err**2))
        d_mae = float(np.mean(d_abs))
        d_max = float(np.max(d_abs))
        d_huber = huber_numpy(d_err, delta=deltaD)
        d_huber_mean = float(np.mean(d_huber))
        d_corr = _safe_corrcoef(d_true, d_pred)
        d_r2 = float(d_corr**2)
        d_thr = np.percentile(d_abs, 95)
    else:
        d_err = d_abs = np.array([])
        d_mse = d_mae = d_max = d_huber_mean = d_corr = d_r2 = d_thr = np.nan

    # --- ANGLES ---
    if nA > 0:
        a_true_rad, ang_labels = _angles_from_sincos_blocks(a_true, n_phi, n_psi)
        a_pred_rad, _          = _angles_from_sincos_blocks(a_pred, n_phi, n_psi)
        a_circ = circular_diff_rad(a_true_rad, a_pred_rad)
        a_mae_rad = float(np.mean(a_circ))
        a_mae_deg = float(np.degrees(a_mae_rad))
        a_max_rad = float(np.max(a_circ))
        a_thr = np.percentile(a_circ, 95)
        a_huber = huber_numpy(a_circ, delta=deltaA)
        a_huber_mean = float(np.mean(a_huber))
    else:
        ang_labels = []
        a_true_rad = a_pred_rad = a_circ = np.array([])
        a_mae_rad = a_mae_deg = a_max_rad = a_thr = a_huber_mean = np.nan

    # --- Weighted Huber ---
    wD = (nD / (nD + nA)) if (nD + nA) > 0 else 0.0
    wA = (nA / (nD + nA)) if (nD + nA) > 0 else 0.0
    overall_weighted_huber = (
        (np.mean(huber_numpy(d_err, delta=deltaD)) if nD > 0 else 0.0) * wD +
        (np.mean(huber_numpy(a_circ, delta=deltaA)) if nA > 0 else 0.0) * wA
    )

    # === PLOTS (3x2) ===
    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    fig.suptitle(f"{title_prefix}Reconstruction Quality (sin/cos blocks)", fontsize=16, y=0.98)

    # (1) Distances: Original vs Reconstructed
    ax = axes[0, 0]
    if nD > 0:
        xD = np.arange(nD)
        ax.plot(xD, d_true, label="Original (dist)", linewidth=2, alpha=0.85)
        ax.plot(xD, d_pred, label="Reconstructed (dist)", linewidth=2, alpha=0.85)
        ax.fill_between(xD, d_true, d_pred, alpha=0.2, color='gray', label='Difference')
        ax.set_title("Distances: Original vs Reconstructed")
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # (2) Distance errors
    ax = axes[0, 1]
    if nD > 0:
        xD = np.arange(nD)
        ax.plot(xD, d_err, linewidth=1.7, alpha=0.9)
        ax.fill_between(xD, 0, d_err, where=(d_abs > d_thr), color='red', alpha=0.35,
                        label=f'|err| > P95 = {d_thr:.4f}')
        ax.axhline(0, color="black", linestyle="--")
        ax.axhline(d_thr, color="red", linestyle=":")
        ax.axhline(-d_thr, color="red", linestyle=":")
        ax.set_title("Distances: Reconstruction Error")
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # (3) Angles: Original vs Reconstructed
    ax = axes[1, 0]
    if nA > 0:
        xA = np.arange(len(a_true_rad))
        ax.plot(xA, a_true_rad, label="Original (ang, rad)", linewidth=2, alpha=0.85)
        ax.plot(xA, a_pred_rad, label="Reconstructed (ang, rad)", linewidth=2, alpha=0.85)
        ax.set_title("Angles: Original vs Reconstructed (radians)")
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # (4) Circular angle errors
    ax = axes[1, 1]
    if nA > 0:
        xA = np.arange(len(a_circ))
        ax.plot(xA, a_circ, linewidth=1.7, alpha=0.9)
        ax.axhline(a_thr, color='red', linestyle=':', label=f'P95 = {a_thr:.4f} rad')
        ax.set_title("Angles: Circular Error |Δ| (rad)")
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # (5) Histogram of distance errors
    ax = axes[2, 0]
    if nD > 0:
        ax.hist(d_err, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(d_err), color="blue", linestyle="-", linewidth=2, label=f"Mean {np.mean(d_err):.4f}")
        ax.axvline(np.median(d_err), color="green", linestyle="-", linewidth=2, label=f"Median {np.median(d_err):.4f}")
        ax.set_title("Distribution of Distance Errors")
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # (6) Histogram of angular errors
    ax = axes[2, 1]
    if nA > 0:
        ax.hist(a_circ, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(a_circ), color="blue", linestyle="-", linewidth=2, label=f"Mean {np.mean(a_circ):.4f} rad")
        ax.axvline(np.median(a_circ), color="green", linestyle="-", linewidth=2, label=f"Median {np.median(a_circ):.4f} rad")
        ax.set_title("Distribution of Angular Errors (rad)")
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # --- Summary metrics ---
    stats_lines = [
        "QUALITY METRICS (sin/cos blocks)",
        f"DISTANCES (n={nD}): MSE={d_mse:.6f}, MAE={d_mae:.6f}, Max={d_max:.6f}, Huber(mean,δ={deltaD})={d_huber_mean:.6f}, R²={d_r2:.3f}",
        f"ANGLES (φ={n_phi}, ψ={n_psi}): MAE={a_mae_rad:.6f} rad ({a_mae_deg:.1f}°), Max={a_max_rad:.6f}, Huber(mean,δ={deltaA})={a_huber_mean:.6f}",
        f"Weighted Huber = {overall_weighted_huber:.6f} [wD={wD:.2f}, wA={wA:.2f}]"
    ]
    fig.text(-0.01, -0.01, "\n".join(stats_lines), fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    metrics = {
        "distances": {"mse": d_mse, "mae": d_mae, "max_abs_error": d_max,
                      "huber_mean": d_huber_mean, "r_squared": d_r2, "correlation": d_corr},
        "angles": {"circ_mae_rad": a_mae_rad, "circ_mae_deg": a_mae_deg,
                   "max_circ_error_rad": a_max_rad, "huber_mean_circ": a_huber_mean},
        "weights": {"distances": wD, "angles": wA},
        "overall_weighted_huber": overall_weighted_huber
    }
    return fig, metrics

# ==============================
# plot_section_errors (distances or angles)
# ==============================
import matplotlib.colors as mcolors

def plot_section_errors(orig, recon, nD, n_phi, n_psi, n_sections=20, kind="distance", title=None):
    orig = _to_numpy(orig); recon = _to_numpy(recon)
    d_true = orig[:nD]; d_pred = recon[:nD]
    a_true = orig[nD:]; a_pred = recon[nD:]

    if kind == "distance":
        values = np.abs(d_true - d_pred)
        total_len = nD
        xlabels_prefix = "D"
        ylabel = "Mean Abs Error"
    elif kind == "angle":
        a_true_rad, _ = _angles_from_sincos_blocks(a_true, n_phi, n_psi)
        a_pred_rad, _ = _angles_from_sincos_blocks(a_pred, n_phi, n_psi)
        values = circular_diff_rad(a_true_rad, a_pred_rad)
        total_len = len(values)
        xlabels_prefix = "A"
        ylabel = "Mean Circular Error (rad)"
    else:
        raise ValueError("kind must be 'distance' or 'angle'.")

    section_size = max(1, total_len // n_sections)
    sections, section_labels, per_section_values = [], [], []
    for i in range(0, total_len, section_size):
        end_idx = min(i + section_size, total_len)
        v = values[i:end_idx]
        sections.append(float(np.mean(v)))
        per_section_values.append(v)
        section_labels.append(f'{xlabels_prefix}{i}-{xlabels_prefix}{end_idx-1}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title or f"Section Errors ({kind})", fontsize=16, y=1.02)

    # Fixed normalization 0-2
    norm = mcolors.Normalize(vmin=0, vmax=2)

    # Bar plot
    colors = plt.cm.Reds(norm(sections))
    bars = ax1.bar(range(len(sections)), sections, color=colors, edgecolor='black')
    for bar, err in zip(bars, sections):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{err:.4f}', ha='center', va='bottom', fontsize=8)
    ax1.set_title("Mean Error per Section")
    ax1.set_ylabel(ylabel)
    ax1.set_xticks(range(len(sections)))
    ax1.set_xticklabels(section_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 2)   # fixed scale

    # Heatmap
    max_len = max(len(sec) for sec in per_section_values)
    error_matrix = [list(sec) + [np.nan]*(max_len-len(sec)) for sec in per_section_values]
    im = ax2.imshow(error_matrix, cmap='Reds', aspect='auto', norm=norm)
    ax2.set_yticks(range(len(section_labels)))
    ax2.set_yticklabels(section_labels)
    ax2.set_title("Detailed Heatmap of Errors")
    plt.colorbar(im, ax=ax2, label=ylabel)

    plt.tight_layout()
    return fig, {"sections": sections, "labels": section_labels, "kind": kind}
