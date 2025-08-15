import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import trustworthiness as skl_trustworthiness

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

primary_color = '#2E86C1'
secondary_color = '#E74C3C'
accent_color = '#F39C12'
bg_color = '#F8F9FA'

# ------------------ Helper: estrazione dati dal tf.data ------------------
def take_from_dataset(ds, max_samples=10000):
    xs = []
    tot = 0
    for batch in ds:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = tf.convert_to_tensor(x)
        xs.append(x)
        tot += int(x.shape[0])
        if tot >= max_samples:
            break
    if not xs:
        raise ValueError("Validation dataset vuoto: controlla ds_val.")
    X = tf.concat(xs, axis=0)
    # appiattisci se rank>2 (es. immagini)
    if len(X.shape) > 2:
        X = tf.reshape(X, [tf.shape(X)[0], -1])
    return X[:max_samples]

# ------------------ Helper: encoder/decoder in batch ---------------------
def run_encoder_decoder(model, X, batch_size=64):
    z_means, z_log_vars, z_samps, recons = [], [], [], []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        z_mean, z_log_var, z = model.encoder(xb, training=False)
        # Per analisi decodifico dal MEAN (più stabile del sample)
        recon = model.decoder(z_mean, training=False)
        z_means.append(z_mean.numpy())
        z_log_vars.append(z_log_var.numpy())
        z_samps.append(z.numpy())
        recons.append(recon.numpy())
    z_mean = np.concatenate(z_means, axis=0)
    z_log_var = np.concatenate(z_log_vars, axis=0)
    z = np.concatenate(z_samps, axis=0)
    X_hat = np.concatenate(recons, axis=0)
    return z_mean, z_log_var, z, X_hat

# ------------------ Metriche ------------------
def kl_per_sample(z_mean, z_log_var):
    # KL(q(z|x) || N(0,I)) per sample (somma sulle dimensioni)
    return -0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1)

def recon_errors(X, X_hat):
    mse = np.mean((X - X_hat) ** 2, axis=1)
    mae = np.mean(np.abs(X - X_hat), axis=1)
    combo = 0.8 * mse + 0.2 * mae
    return mse, mae, combo

def corr_offdiag(cov):
    d = cov.shape[0]
    std = np.sqrt(np.diag(cov) + 1e-12)
    corr = cov / (std[:, None] * std[None, :] + 1e-12)
    off = corr.copy()
    np.fill_diagonal(off, np.nan)
    return corr, off[~np.isnan(off)]

def safe_trustworthiness(X, Z, k):
    n = X.shape[0]
    k_eff = min(k, max(1, int(n/2 - 1)))
    return float(skl_trustworthiness(X, Z, n_neighbors=k_eff))

def create_summary_page(pdf, X_val, latent_dim, beta_val, cov_reg_val, 
                       kl_mean, kl_ps, recon_mse, recon_mae, recon_combo,
                       mu_mean, mu_var, off_diag, tw_k):
    """Crea una pagina di sommario più professionale"""

    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Header principale
    fig.suptitle("β-VAE Latent Space Analysis Report", fontsize=18, fontweight='bold', y=0.95)
    
    # Data e timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.85, 0.92, f"Generated: {timestamp}", ha='right', fontsize=8, style='italic')
    
    # Sezioni organizzate
    sections = []
    
    # 1. Configurazione del modello
    sections.append("📊 MODEL CONFIGURATION")
    sections.append(f"   Input dimension:     {X_val.shape[1]:,}")
    sections.append(f"   Latent dimension:    {latent_dim}")
    sections.append(f"   Validation samples:  {X_val.shape[0]:,}")
    sections.append(f"   β parameter:         {beta_val if beta_val is not None else 'N/A'}")
    sections.append(f"   Covariance reg:      {cov_reg_val if cov_reg_val is not None else 'N/A'}")
    sections.append("")
    
    # 2. Metriche KL
    sections.append("🔥 KL DIVERGENCE METRICS")
    sections.append(f"   Mean KL (unweighted): {kl_mean:.4f}")
    sections.append(f"   KL std deviation:     {np.std(kl_ps):.4f}")
    sections.append(f"   KL min/max:          [{np.min(kl_ps):.4f}, {np.max(kl_ps):.4f}]")
    sections.append("")
    
    # 3. Metriche di ricostruzione
    sections.append("🔧 RECONSTRUCTION METRICS")
    sections.append(f"   MSE:                 {recon_mse:.6f}")
    sections.append(f"   MAE:                 {recon_mae:.6f}")
    sections.append(f"   Combined (0.8*MSE+0.2*MAE): {recon_combo:.6f}")
    sections.append("")
    
    # 4. Statistiche spazio latente
    sections.append("🎯 LATENT SPACE STATISTICS")
    sections.append(f"   z_mean statistics:")
    sections.append(f"     • Mean: {np.array2string(mu_mean, precision=3, suppress_small=True)}")
    sections.append(f"     • Var:  {np.array2string(mu_var, precision=3, suppress_small=True)}")
    sections.append(f"   Avg |correlation| off-diagonal: {np.nanmean(np.abs(off_diag)):.4f}")
    sections.append("")
    
    # 5. Metriche di qualità
    sections.append("✅ EMBEDDING QUALITY")
    sections.append("   Trustworthiness scores:")
    for k in sorted(tw_k.keys()):
        quality_emoji = "🟢" if tw_k[k] > 0.9 else "🟡" if tw_k[k] > 0.7 else "🔴"
        sections.append(f"     • k={k:2d}: {tw_k[k]:.4f} {quality_emoji}")
    
    # Rendering del testo con formattazione
    y_pos = 0.85
    for line in sections:
        if line.startswith(("📊", "🔥", "🔧", "🎯", "✅")):

            fig.text(0.08, y_pos, line, fontsize=12, fontweight='bold', 
                    color=primary_color, va='top')
            y_pos -= 0.04
        elif line == "":
            y_pos -= 0.02
        else:
            # Contenuto normale
            fig.text(0.08, y_pos, line, fontsize=10, fontfamily='monospace', 
                    va='top', color='#2C3E50')
            y_pos -= 0.025
    
    # Footer con note
    footer_text = ("Note: Trustworthiness measures how well the latent space preserves "
                  "local neighborhood structure from the original space.")
    fig.text(0.08, 0.05, footer_text, fontsize=8, style='italic', 
             wrap=True, va='bottom', color='#7F8C8D')
    
    pdf.savefig(fig, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_latent_scatter(pdf, z_mean, combo_ps, latent_dim):
    """Scatter plot migliorato dello spazio latente"""
    if latent_dim < 2:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Latent Space Visualization", fontsize=14, fontweight='bold')
    
    # Plot 1: Colorato per errore di ricostruzione
    scatter = ax1.scatter(z_mean[:, 0], z_mean[:, 1], 
                         c=combo_ps, s=15, alpha=0.7, 
                         cmap='RdYlBu_r', edgecolors='none')
    ax1.set_title("Colored by Reconstruction Error")
    ax1.set_xlabel("Latent Dimension 1")
    ax1.set_ylabel("Latent Dimension 2")
    ax1.grid(True, alpha=0.3)
    
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label("0.8×MSE + 0.2×MAE", rotation=270, labelpad=15)
    
    # Plot 2: Densità con contorni
    from scipy.stats import gaussian_kde
    if len(z_mean) > 100:  # Solo se abbiamo abbastanza campioni
        xy = np.vstack([z_mean[:, 0], z_mean[:, 1]])
        kde = gaussian_kde(xy)
        
        # Griglia per la densità
        x_min, x_max = z_mean[:, 0].min() - 1, z_mean[:, 0].max() + 1
        y_min, y_max = z_mean[:, 1].min() - 1, z_mean[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)
        
        ax2.contourf(xx, yy, density, levels=15, cmap='Blues', alpha=0.8)
        ax2.scatter(z_mean[:, 0], z_mean[:, 1], s=8, alpha=0.5, c='darkblue')
    else:
        # Fallback per pochi campioni
        h = ax2.hist2d(z_mean[:, 0], z_mean[:, 1], bins=30, cmap='Blues')
        plt.colorbar(h[3], ax=ax2)
    
    ax2.set_title("Density Distribution")
    ax2.set_xlabel("Latent Dimension 1")
    ax2.set_ylabel("Latent Dimension 2")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_dimension_analysis(pdf, z_mean, latent_dim):
    """Analisi dettagliata delle dimensioni latenti"""
    max_dims = min(8, latent_dim)  # Mostra fino a 8 dimensioni
    
    if max_dims <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.flatten()
    
    fig.suptitle("Latent Dimensions Analysis", fontsize=14, fontweight='bold')
    
    for i in range(max_dims):
        ax = axes[i]
        
        # Istogramma con curva di densità
        n, bins, patches = ax.hist(z_mean[:, i], bins=40, alpha=0.7, 
                                  color=primary_color, density=True)
        
        # Aggiungi statistiche
        mean_val = np.mean(z_mean[:, i])
        std_val = np.std(z_mean[:, i])
        
        # Linea della media
        ax.axvline(mean_val, color=secondary_color, linestyle='--', 
                  linewidth=2, alpha=0.8, label=f'μ={mean_val:.3f}')
        
        # Linee di ±1 std
        ax.axvline(mean_val + std_val, color=accent_color, linestyle=':', 
                  alpha=0.7, label=f'σ={std_val:.3f}')
        ax.axvline(mean_val - std_val, color=accent_color, linestyle=':', alpha=0.7)
        
        ax.set_title(f'Dimension {i+1}', fontweight='bold')
        ax.set_xlabel(f'z_{i+1}')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Nascondi assi non utilizzati
    for i in range(max_dims, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_correlation_analysis(pdf, mu_corr, latent_dim):
    """Analisi di correlazione migliorata"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Latent Space Correlation Analysis", fontsize=14, fontweight='bold')
    
    # Matrice di correlazione
    im1 = ax1.imshow(mu_corr, cmap='RdBu_r', vmin=-1, vmax=1, 
                     interpolation='nearest')
    ax1.set_title("Correlation Matrix")
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("Latent Dimension")
    
    # Aggiungi numeri nella matrice se piccola
    if latent_dim <= 10:
        for i in range(latent_dim):
            for j in range(latent_dim):
                text = ax1.text(j, i, f'{mu_corr[i, j]:.2f}',
                               ha="center", va="center", 
                               color="white" if abs(mu_corr[i, j]) > 0.5 else "black",
                               fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='Correlation')
    
    # Distribuzione delle correlazioni off-diagonali
    off_diag_vals = mu_corr[np.triu_indices_from(mu_corr, k=1)]
    ax2.hist(off_diag_vals, bins=30, alpha=0.7, color=primary_color, 
             edgecolor='black', linewidth=0.5)
    ax2.axvline(np.mean(off_diag_vals), color=secondary_color, 
                linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(off_diag_vals):.3f}')
    ax2.set_title("Off-diagonal Correlations Distribution")
    ax2.set_xlabel("Correlation Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_metrics_dashboard(pdf, combo_ps, kl_ps, tw_k):
    """Dashboard delle metriche principali"""
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    fig.suptitle("Performance Metrics Dashboard", fontsize=14, fontweight='bold')
    
    # 1. Distribuzione errori di ricostruzione
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(combo_ps, bins=40, alpha=0.7, color=primary_color, edgecolor='black')
    ax1.axvline(np.mean(combo_ps), color=secondary_color, linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(combo_ps):.4f}')
    ax1.axvline(np.median(combo_ps), color=accent_color, linestyle=':', 
                linewidth=2, label=f'Median: {np.median(combo_ps):.4f}')
    ax1.set_title("Reconstruction Error")
    ax1.set_xlabel("0.8×MSE + 0.2×MAE")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribuzione KL
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(kl_ps, bins=40, alpha=0.7, color=accent_color, edgecolor='black')
    ax2.axvline(np.mean(kl_ps), color=secondary_color, linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(kl_ps):.4f}')
    ax2.set_title("KL Divergence (per sample)")
    ax2.set_xlabel("KL (nats)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trustworthiness scores
    ax3 = fig.add_subplot(gs[0, 2])
    ks = sorted(tw_k.keys())
    vals = [tw_k[k] for k in ks]
    colors = ['#27AE60' if v > 0.9 else '#F39C12' if v > 0.7 else '#E74C3C' for v in vals]
    bars = ax3.bar([str(k) for k in ks], vals, color=colors, alpha=0.8)
    ax3.set_ylim(0, 1.0)
    ax3.set_title("Trustworthiness Scores")
    ax3.set_xlabel("k neighbors")
    ax3.set_ylabel("Score")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi valori sopra le barre
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Box plot comparativo (errori vs KL)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Normalizza per confronto visivo
    combo_norm = (combo_ps - np.min(combo_ps)) / (np.max(combo_ps) - np.min(combo_ps))
    kl_norm = (kl_ps - np.min(kl_ps)) / (np.max(kl_ps) - np.min(kl_ps))
    
    box_data = [combo_norm, kl_norm]
    box_labels = ['Reconstruction Error\n(normalized)', 'KL Divergence\n(normalized)']
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor(primary_color)
    bp['boxes'][1].set_facecolor(accent_color)
    
    ax4.set_title("Normalized Metrics Comparison")
    ax4.set_ylabel("Normalized Value")
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight', facecolor='white')
    plt.close(fig)