import tensorflow as tf
from src.b_implement import BetaVAEMonitor, KLThresholdCallback


class CovRegScheduler(tf.keras.callbacks.Callback):
    """
    Turns on and linearly ramps 'cov_reg' from 0 -> target_cov_reg once KL_ema
    stays inside the deadband for `in_band_epochs` epochs.

    This avoids fighting the β-controller early on.
    """
    def __init__(self,
                 target_cov_reg: float = 1e-3,
                 ramp_epochs: int = 10,
                 in_band_epochs: int = 5,
                 target_kl: float = 5.0,
                 deadband: float = 0.20,
                 ema_alpha: float = 0.10,
                 kl_key: str = "val_kl_loss_unweighted",
                 print_every: int = 5):
        super().__init__()
        self.target_cov_reg = float(target_cov_reg)
        self.ramp_epochs = int(ramp_epochs)
        self.in_band_epochs = int(in_band_epochs)
        self.target_kl = float(target_kl)
        self.deadband = float(deadband)
        self.ema_alpha = float(ema_alpha)
        self.kl_key = kl_key
        self.print_every = int(print_every)

        self._ema = None
        self._inband_count = 0
        self._ramping = False
        self._ramp_step = 0

    def on_train_begin(self, logs=None):
        # start turned off
        self.model.cov_reg.assign(0.0)

    def _update_ema(self, val):
        self._ema = val if self._ema is None else (1 - self.ema_alpha) * self._ema + self.ema_alpha * val

    def _in_band(self):
        low  = self.target_kl * (1 - self.deadband)
        high = self.target_kl * (1 + self.deadband)
        return (self._ema is not None) and (low <= self._ema <= high)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        kl = logs.get(self.kl_key, None)
        if kl is None:
            kl = logs.get("kl_loss_unweighted", None)
        if kl is None:
            if (epoch + 1) % self.print_every == 0:
                print(f"[CovRegScheduler] KL key '{self.kl_key}' not found; cov_reg={float(self.model.cov_reg.numpy()):.2e}")
            return

        kl = float(kl)
        self._update_ema(kl)

        if not self._ramping:
            if self._in_band():
                self._inband_count += 1
            else:
                self._inband_count = 0

            if self._inband_count >= self.in_band_epochs:
                self._ramping = True
                self._ramp_step = 0
                print(f"[CovRegScheduler] KL_ema in band → start ramp to {self.target_cov_reg:.2e} "
                      f"in {self.ramp_epochs} epochs.")

        if self._ramping:
            frac = min(1.0, (self._ramp_step + 1) / max(1, self.ramp_epochs))
            new_w = frac * self.target_cov_reg
            self.model.cov_reg.assign(new_w)
            self._ramp_step += 1

        if (epoch + 1) % self.print_every == 0:
            print(f"[CovRegScheduler] ep {epoch+1}  KL_ema={self._ema:.4f}  cov_reg={float(self.model.cov_reg.numpy()):.2e}")

def callbacks(log_dir, latent_dim, monitor="val_loss", model='vae'):
    cb = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=15,
            min_delta=1e-6,
            restore_best_weights=True,
            verbose=1,
            mode="min"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    if model == 'ae':
        filepath = f'ae_{latent_dim}d.keras'
    else:
        filepath = f'vae_{latent_dim}d.keras'

    cb.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    )

    if model == 'vae':
        #cb.append(BetaVAEMonitor())
        cb.append(KLThresholdCallback(
        target_kl=5.0,                 # ≈2.5 nats/dim
        adjustment_rate=0.06,          # passo di correzione su β (±6%)
        deadband=0.18,                 # banda morta: 5*(1±0.18) ⇒ [4.10, 5.90]
        ema_alpha=0.10,                # smoothing della KL
        beta_min=1e-4,
        beta_max=1.2e-3,               # alza il “tetto” per far scendere la KL se resta alta
        kl_key="val_kl_loss_unweighted"
    ))
        '''
        cb.append(CovRegScheduler(
        target_cov_reg=7e-4,    # try 3e-4 to 1e-3; back off if val recon worsens
        ramp_epochs=10,
        in_band_epochs=5,
        target_kl=5.0,          # keep consistent with the β-controller
        deadband=0.20,
        ema_alpha=0.10,
        kl_key="val_kl_loss_unweighted",
        print_every=5
        ))
        '''
    return cb