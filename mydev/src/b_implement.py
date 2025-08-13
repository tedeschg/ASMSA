import tensorflow as tf
import math


class BetaVAEMonitor(tf.keras.callbacks.Callback):
    """Monitora e stampa le metriche della Beta-VAE ogni N epoche."""
    
    def __init__(self, print_every=10):
        super().__init__()
        self.print_every = print_every
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every == 0:
            kl_loss = logs.get('kl_loss', 0)
            recon_loss = logs.get('reconstruction_loss', 0)
            val_kl_loss = logs.get('val_kl_loss', 0)
            val_recon_loss = logs.get('val_reconstruction_loss', 0)
            
            print(f"\nEpoca {epoch+1}: Beta={self.model.beta:.4f}")
            print(f"  Train - KL={kl_loss:.6f}, Recon={recon_loss:.6f}")
            if val_kl_loss > 0 or val_recon_loss > 0:
                print(f"  Val   - KL={val_kl_loss:.6f}, Recon={val_recon_loss:.6f}")


class BetaWarmupCallback(tf.keras.callbacks.Callback):
    """Implementa il warmup graduale del parametro beta."""
    
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
        if epoch < self.warmup_epochs:
            # frazione di warmup completata
            p = float(epoch) / self.warmup_epochs
            # interpolazione lineare tra beta_start e beta_target
            new_beta = self.beta_start + p * (self.beta_target - self.beta_start)
            self.model.beta = new_beta
            print(f"→ Epoca {epoch+1}: beta = {new_beta:.6f}")
        else:
            # assicurati che beta sia esattamente il target dopo il warmup
            if self.model.beta != self.beta_target:
                self.model.beta = self.beta_target
                print(f"→ Epoca {epoch+1}: beta = {self.beta_target:.6f} (target raggiunto)")
