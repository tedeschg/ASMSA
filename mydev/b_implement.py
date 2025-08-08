import tensorflow as tf

class BetaVAEMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            kl_loss = logs.get('kl_loss', 0)
            recon_loss = logs.get('reconstruction_loss', 0)
            print(f"\nEpoca {epoch+1}: Beta={self.model.beta:.4f}, "
                  f"KL={kl_loss:.6f}, Recon={recon_loss:.6f}")

class BetaWarmupCallback(tf.keras.callbacks.Callback):
    def __init__(self, beta_target, warmup_epochs):
        super().__init__()
        self.beta_target = beta_target
        self.warmup_epochs = warmup_epochs
        self.beta_start = None  # verrà popolato in on_train_begin

    def on_train_begin(self, logs=None):
        # prende il beta iniziale direttamente dal modello
        self.beta_start = float(self.model.beta)
        print(f"Warmup: start from beta = {self.beta_start:.6f} "
              f"until beta = {self.beta_target:.6f} in {self.warmup_epochs} epoche.")

    def on_epoch_begin(self, epoch, logs=None):
        # frazione di warmup completata
        p = min(1.0, float(epoch) / self.warmup_epochs)
        # interpolazione lineare tra beta_start e beta_target
        new_beta = self.beta_start + p * (self.beta_target - self.beta_start)
        self.model.beta = new_beta
        print(f"→ Epoca {epoch+1}: beta = {new_beta:.6f}")