import tensorflow as tf

class BetaAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, beta_min=1e-3, beta_max=3.0, n_epochs=50):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_epochs = n_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # progress lineare da 0 a 1
        progress = min(epoch / self.n_epochs, 1.0)
        beta = self.beta_min + (self.beta_max - self.beta_min) * progress

        # Assumendo che il modello abbia un attributo `beta` usato nella loss
        self.model.beta = tf.constant(beta, dtype=tf.float32)

        print(f"Epoch {epoch:03d} → beta={beta:.6f}")

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

