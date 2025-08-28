import tensorflow as tf
from src.b_implement import BetaVAEMonitor, BetaAnnealingCallback

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
        cb.append(BetaVAEMonitor())
        cb.append(BetaAnnealingCallback(beta_min=0.001, beta_max=2, n_epochs=50))
    return cb