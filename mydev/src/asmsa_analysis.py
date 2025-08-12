import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Configurazione stile matplotlib
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def analyze_reconstruction(orig, recon, feature_names=None, title_prefix=""):
    """
    Analizza e visualizza la qualità della ricostruzione
    
    Args:
        orig: array originale
        recon: array ricostruito
        feature_names: lista nomi delle feature (opzionale)
        title_prefix: prefisso per i titoli
    """
    # Converti in numpy se necessario
    if hasattr(orig, 'numpy'):
        orig = orig.numpy()
    if hasattr(recon, 'numpy'):
        recon = recon.numpy()
    
    # Calcoli metriche
    error = orig - recon
    abs_error = np.abs(error)
    mse = np.mean(error**2)
    mae = np.mean(abs_error)
    max_error = np.max(abs_error)
    
    # Soglie per evidenziare errori significativi
    error_threshold = np.percentile(abs_error, 95)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{title_prefix}Analisi Qualità Ricostruzione', fontsize=16, y=0.96)
    
    # 1️⃣ Confronto originale vs ricostruito
    ax1 = axes[0, 0]
    x_axis = np.arange(len(orig))
    ax1.plot(x_axis, orig, label="Originale", linewidth=2, alpha=0.8, color='#2E86AB')
    ax1.plot(x_axis, recon, label="Ricostruito", linewidth=2, alpha=0.8, color='#A23B72')
    ax1.fill_between(x_axis, orig, recon, alpha=0.2, color='gray', label='Differenza')
    ax1.set_title("Confronto Serie Temporali")
    ax1.set_xlabel("Feature Index")
    ax1.set_ylabel("Valore")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2️⃣ Errore di ricostruzione con zone critiche
    ax2 = axes[0, 1]
    ax2.plot(x_axis, error, color='#F18F01', linewidth=1.5, alpha=0.8)
    ax2.fill_between(x_axis, 0, error, where=(abs_error > error_threshold), 
                     color='red', alpha=0.4, label=f'Errori > {error_threshold:.3f}')
    ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(error_threshold, color="red", linestyle=":", alpha=0.7)
    ax2.axhline(-error_threshold, color="red", linestyle=":", alpha=0.7)
    ax2.set_title("Errore di Ricostruzione")
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Errore")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3️⃣ Distribuzione errori
    ax3 = axes[1, 0]
    ax3.hist(error, bins=30, alpha=0.7, color='#C73E1D', edgecolor='black', linewidth=0.5)
    ax3.axvline(0, color="black", linestyle="--", linewidth=2)
    ax3.axvline(np.mean(error), color="blue", linestyle="-", linewidth=2, label=f'Media: {np.mean(error):.4f}')
    ax3.axvline(np.median(error), color="green", linestyle="-", linewidth=2, label=f'Mediana: {np.median(error):.4f}')
    ax3.set_title("Distribuzione Errori")
    ax3.set_xlabel("Errore")
    ax3.set_ylabel("Frequenza")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Aggiunta spiegazione nella distribuzione
    ax3.text(0.02, 0.98, 
             '💡 Cosa cercare:\n'
             '✅ Centrato su 0 (no bias)\n'
             '✅ Forma simmetrica\n'
             '❌ Spostato = bias sistematico\n'
             '❌ Code lunghe = outlier', 
             transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
             verticalalignment='top', fontsize=9)
    
    # 4️⃣ Scatter plot con correlazione
    ax4 = axes[1, 1]
    scatter = ax4.scatter(orig, recon, alpha=0.6, c=abs_error, cmap='Reds', s=30)
    
    # Linea ideale (y=x)
    min_val, max_val = min(orig.min(), recon.min()), max(orig.max(), recon.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8, label='Ricostruzione Perfetta')
    
    # Calcola R²
    correlation = np.corrcoef(orig, recon)[0, 1]
    r_squared = correlation**2
    
    ax4.set_title(f"Correlazione Orig-Ricostr (R² = {r_squared:.4f})")
    ax4.set_xlabel("Valori Originali")
    ax4.set_ylabel("Valori Ricostruiti")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Colorbar per scatter
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Errore Assoluto')
    
    plt.tight_layout()
    
    # Box con statistiche
    stats_text = f"""METRICHE QUALITÀ:
MSE: {mse:.6f}
MAE: {mae:.6f}
Max Error: {max_error:.6f}
R²: {r_squared:.6f}
Std Error: {np.std(error):.6f}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    return fig, {
        'mse': mse,
        'mae': mae, 
        'max_error': max_error,
        'r_squared': r_squared,
        'correlation': correlation,
        'std_error': np.std(error)
    }

def plot_section_errors(orig, recon, n_sections=20, title="Errori per Sezione"):
    """
    Visualizza solo gli errori per sezioni delle feature
    
    Args:
        orig: array originale  
        recon: array ricostruito
        n_sections: numero di sezioni da creare
        title: titolo del grafico
    """
    if hasattr(orig, 'numpy'):
        orig = orig.numpy()
    if hasattr(recon, 'numpy'):
        recon = recon.numpy()
    
    abs_errors = np.abs(orig - recon)
    
    # Calcola dimensione sezione automatica
    section_size = max(1, len(orig) // n_sections)
    
    sections = []
    section_errors = []
    section_labels = []
    
    for i in range(0, len(orig), section_size):
        end_idx = min(i + section_size, len(orig))
        section_error = np.mean(abs_errors[i:end_idx])
        sections.append(section_error)
        section_errors.append(abs_errors[i:end_idx])
        section_labels.append(f'{i}-{end_idx-1}')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # 1️⃣ Grafico a barre degli errori medi per sezione
    colors = plt.cm.Reds(np.array(sections) / max(sections))
    bars = ax1.bar(range(len(sections)), sections, color=colors, edgecolor='black', linewidth=0.5)
    
    # Aggiungi valori sopra le barre
    for i, (bar, error) in enumerate(zip(bars, sections)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{error:.4f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Linea media globale
    global_mean = np.mean(abs_errors)
    ax1.axhline(global_mean, color='blue', linestyle='--', linewidth=2, 
                label=f'Errore Medio Globale: {global_mean:.4f}')
    
    ax1.set_xticks(range(0, len(sections), max(1, len(sections)//10)))
    ax1.set_xticklabels([section_labels[i] for i in range(0, len(sections), max(1, len(sections)//10))], 
                       rotation=45, ha='right')
    ax1.set_title('Errore Medio per Sezione')
    ax1.set_xlabel('Range Feature')
    ax1.set_ylabel('Errore Assoluto Medio')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2️⃣ Heatmap bidimensionale degli errori
    # Riorganizza in matrice per heatmap più dettagliata
    max_len = max(len(section) for section in section_errors)
    error_matrix = []
    
    for section in section_errors:
        # Riempi sezioni più corte con NaN per una visualizzazione corretta
        padded_section = list(section) + [np.nan] * (max_len - len(section))
        error_matrix.append(padded_section)
    
    error_matrix = np.array(error_matrix)
    
    # Crea heatmap
    im = ax2.imshow(error_matrix, cmap='Reds', aspect='auto', interpolation='nearest')
    
    # Impostazioni assi
    ax2.set_yticks(range(0, len(sections), max(1, len(sections)//10)))
    ax2.set_yticklabels([section_labels[i] for i in range(0, len(sections), max(1, len(sections)//10))])
    ax2.set_xlabel('Posizione nella Sezione')
    ax2.set_ylabel('Sezione Feature')
    ax2.set_title('Heatmap Dettagliata Errori per Sezione')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, label='Errore Assoluto')
    
    plt.tight_layout()
    
    # Statistiche per sezione
    stats_text = f"""STATISTICHE SEZIONI:
Sezioni Totali: {len(sections)}
Sezione Peggiore: {section_labels[np.argmax(sections)]} (Errore: {max(sections):.4f})
Sezione Migliore: {section_labels[np.argmin(sections)]} (Errore: {min(sections):.4f})
Std tra Sezioni: {np.std(sections):.4f}
Range Errori: {max(sections) - min(sections):.4f}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8))
    
    return fig, {
        'section_errors': sections,
        'section_labels': section_labels,
        'worst_section': section_labels[np.argmax(sections)],
        'best_section': section_labels[np.argmin(sections)],
        'sections_std': np.std(sections)
    }