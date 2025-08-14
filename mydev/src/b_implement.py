import tensorflow as tf

import math
import tensorflow as tf

class KLThresholdCallback(tf.keras.callbacks.Callback):
    """
    Keeps the *unweighted* KL (nats per sample) near a target by adjusting β.
    Uses an EMA of KL and a deadband around the target to avoid jitter.

    - If KL_ema < low  => increase capacity => decrease β (down to beta_min)
    - If KL_ema > high => decrease capacity => increase β (up to beta_max)
    - Else (within deadband): leave β unchanged

    Notes:
      * `kl_key` should be the unweighted KL you log, e.g. 'val_kl_loss_unweighted'.
      * β is expected to be a non-trainable tf.Variable on the model.
    """
    def __init__(self,
                 target_kl: float,
                 adjustment_rate: float = 0.05,
                 deadband: float = 0.20,
                 ema_alpha: float = 0.10,
                 beta_min: float = 1e-4,
                 beta_max: float = 1e-3,
                 kl_key: str = "val_kl_loss_unweighted",
                 print_every: int = 5):
        super().__init__()
        self.target_kl = float(target_kl)
        self.adjustment_rate = float(adjustment_rate)
        self.deadband = float(deadband)
        self.ema_alpha = float(ema_alpha)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.kl_key = kl_key
        self.print_every = int(print_every)
        self._ema = None

    def _update_ema(self, val):
        self._ema = val if self._ema is None else (1 - self.ema_alpha) * self._ema + self.ema_alpha * val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        kl = logs.get(self.kl_key, None)
        if kl is None:
            kl = logs.get("kl_loss_unweighted", None)  # fallback to train KL if val missing
        if kl is None:
            if (epoch + 1) % self.print_every == 0:
                print(f"[KLThresholdCallback] KL key '{self.kl_key}' not found in logs.")
            return

        kl = float(kl)
        self._update_ema(kl)

        low  = self.target_kl * (1 - self.deadband)
        high = self.target_kl * (1 + self.deadband)

        beta_now = float(self.model.beta.numpy())
        action = "="

        if self._ema < low:
            # KL too low -> increase capacity -> decrease β
            beta_new = max(self.beta_min, beta_now * (1.0 - self.adjustment_rate))
            action = "β↓"
        elif self._ema > high:
            # KL too high -> decrease capacity -> increase β
            beta_new = min(self.beta_max, beta_now * (1.0 + self.adjustment_rate))
            action = "β↑"
        else:
            beta_new = beta_now

        if beta_new != beta_now:
            self.model.beta.assign(beta_new)

        if (epoch + 1) % self.print_every == 0:
            print(f"[KLThresholdCallback] ep {epoch+1}  KL={kl:.4f}  EMA={self._ema:.4f}  "
                  f"target={self.target_kl:.2f}  β={self.model.beta.numpy():.6f}  {action}")



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

