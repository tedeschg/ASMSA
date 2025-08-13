import tensorflow as tf

class KLThresholdCallback(tf.keras.callbacks.Callback):
    """
    Regola β per mantenere la KL (NON pesata) vicino a un target.
    - EMA per ridurre jitter
    - Deadband per evitare correzioni inutili
    - Usa tf.Variable.assign per aggiornare β
    """
    def __init__(self,
                 target_kl,
                 adjustment_rate=0.07,
                 deadband=0.2,
                 ema_alpha=0.1,
                 beta_min=1e-4,
                 beta_max=6e-3,
                 kl_key="kl_loss_unweighted"):  # puoi mettere "val_kl_loss_unweighted"
        super().__init__()
        self.target_kl = float(target_kl)
        self.rate      = float(adjustment_rate)
        self.deadband  = float(deadband)
        self.ema_alpha = float(ema_alpha)
        self.beta_min  = float(beta_min)
        self.beta_max  = float(beta_max)
        self.kl_key    = kl_key
        self._kl_ema   = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.kl_key not in logs:
            return

        kl = float(logs[self.kl_key])
        # EMA della KL
        self._kl_ema = kl if self._kl_ema is None else (1 - self.ema_alpha) * self._kl_ema + self.ema_alpha * kl

        low  = self.target_kl * (1 - self.deadband)
        high = self.target_kl * (1 + self.deadband)

        beta_val = float(self.model.beta.numpy())
        if self._kl_ema < low:
            beta_val *= (1 - self.rate)   # KL bassa -> β↓ (così la KL sale)
            action = "β↓"
        elif self._kl_ema > high:
            beta_val *= (1 + self.rate)   # KL alta  -> β↑ (così la KL scende)
            action = "β↑"
        else:
            action = "β="

        beta_val = min(self.beta_max, max(self.beta_min, beta_val))
        self.model.beta.assign(beta_val)

        if (epoch + 1) % 5 == 0:
            print(f"KL(unw)={kl:.4f} | EMA={self._kl_ema:.4f} | target≈{self.target_kl:.2f} → {action} β={self.model.beta.numpy():.6f}")


class BetaVAEMonitor(tf.keras.callbacks.Callback):
    """Stampa ogni 5 epoche: KL (unweighted/pesata), ricostruzione e β corrente."""
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 != 0:
            return
        logs = logs or {}
        kl_w        = logs.get("kl_loss", 0.0)
        kl_unw      = logs.get("kl_loss_unweighted", 0.0)
        recon       = logs.get("reconstruction_loss", 0.0)
        val_kl_unw  = logs.get("val_kl_loss_unweighted", 0.0)
        val_recon   = logs.get("val_reconstruction_loss", 0.0)
        beta_now    = float(self.model.beta.numpy())

        print(f"\n📊 Epoca {epoch+1}: β={beta_now:.6f}")
        print(f"  • Train  → KL(unw): {kl_unw:.4f} | KL(β·): {kl_w:.6f} | Recon: {recon:.6f}")
        print(f"  • Val    → KL(unw): {val_kl_unw:.4f} | Recon: {val_recon:.6f}")

