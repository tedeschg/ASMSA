import tensorflow as tf
from tensorflow.keras import layers, models, metrics

# ==== Sampling e blocco denso ====
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.random.normal((batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def asmsa_block(x, neurons, activation, name_prefix):
    x = layers.Dense(neurons, name=f"{name_prefix}_dense")(x)
    x = layers.BatchNormalization(momentum=0.8, name=f"{name_prefix}_bn")(x)
    x = layers.Activation(activation, name=f"{name_prefix}_act")(x)
    x = layers.Dropout(0.1, name=f"{name_prefix}_dropout")(x)
    return x


class BetaVAE(models.Model):
    """
    Beta-VAE with:
      - β as a non-trainable tf.Variable you can adjust from callbacks
      - Optional 'cov_reg' whitening regularizer (mean->0, covariance->I)
      - Custom training loop (no `loss=` in compile)
      - Logged metrics: total loss, reconstruction, KL (unweighted and β-weighted),
        whitening regularizer value, β, and current cov_reg weight

    You can plug any reconstruction loss by passing `recon_fn`:
        recon_fn(y_true, y_pred) -> per-sample loss (shape (B,)) or a scalar.
    """
    def __init__(self, encoder, decoder,
                 recon_loss_weight=1.0,
                 beta=0.1,
                 cov_reg=0.0,
                 recon_fn=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_weight = float(recon_loss_weight)
        self.recon_fn = recon_fn

        # Non-trainable variables so callbacks can adjust them
        self.beta    = tf.Variable(float(beta),    trainable=False, dtype=tf.float32, name="beta")
        self.cov_reg = tf.Variable(float(cov_reg), trainable=False, dtype=tf.float32, name="cov_reg")

        # Trackers / metrics
        self.total_loss_tracker          = metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_unw_tracker              = metrics.Mean(name="kl_loss_unweighted")  # nats per sample
        self.kl_w_tracker                = metrics.Mean(name="kl_loss")             # β * KL
        self.whiten_reg_tracker          = metrics.Mean(name="whiten_reg")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_unw_tracker,
            self.kl_w_tracker,
            self.whiten_reg_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def _compute_losses(self, x, reconstruction, z_mean, z_log_var):
        # --- Reconstruction loss ---
        if self.recon_fn is not None:
            recon_raw = self.recon_fn(x, reconstruction)
            # Accept (B,) or anything reducible; fall back to mean over non-batch dims.
            if isinstance(recon_raw, tf.Tensor) and recon_raw.shape.rank is not None and recon_raw.shape.rank >= 2:
                recon_per_sample = tf.reduce_mean(recon_raw, axis=list(range(1, recon_raw.shape.rank)))
            else:
                # (B,) or scalar
                recon_per_sample = recon_raw
            recon_loss = tf.reduce_mean(recon_per_sample) * self.recon_loss_weight
        else:
            # Fallback: per-sample MSE/MAE blend
            mse = tf.reduce_mean(tf.square(x - reconstruction), axis=1)  # (B,)
            mae = tf.reduce_mean(tf.abs(x - reconstruction),    axis=1)  # (B,)
            recon_loss = tf.reduce_mean(0.8 * mse + 0.2 * mae) * self.recon_loss_weight

        # --- Unweighted KL (nats per sample): sum over latent dims, mean over batch ---
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        kl_unw = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        # --- β-weighted KL ---
        kl_w = self.beta * kl_unw

        # --- Whitening regularizer (always computed; scaled by cov_reg) ---
        mu = tf.cast(z_mean, tf.float32)                 # (B, d)
        b  = tf.cast(tf.shape(mu)[0], tf.float32)
        d  = tf.shape(mu)[1]

        mean_mu  = tf.reduce_mean(mu, axis=0)           # (d,)
        mean_pen = tf.reduce_sum(tf.square(mean_mu))    # scalar

        mu_centered = mu - mean_mu
        denom = tf.maximum(b - 1.0, 1.0)                # avoid div/0 when batch=1
        cov = tf.matmul(mu_centered, mu_centered, transpose_a=True) / denom  # (d, d)
        eye = tf.eye(d, dtype=tf.float32)
        cov_pen = tf.reduce_mean(tf.square(cov - eye))  # scalar

        reg = tf.cast(self.cov_reg, tf.float32) * (cov_pen + mean_pen)

        total = recon_loss + kl_w + reg
        return total, recon_loss, kl_unw, kl_w, reg

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=True)
            reconstruction       = self.decoder(z, training=True)
            total, recon, kl_unw, kl_w, reg = self._compute_losses(
                x, reconstruction, z_mean, z_log_var
            )

        grads = tape.gradient(total, self.trainable_weights)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(recon)
        self.kl_unw_tracker.update_state(kl_unw)
        self.kl_w_tracker.update_state(kl_w)
        self.whiten_reg_tracker.update_state(reg)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss_unweighted": self.kl_unw_tracker.result(),
            "kl_loss": self.kl_w_tracker.result(),
            "whiten_reg": self.whiten_reg_tracker.result(),
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        z_mean, z_log_var, z = self.encoder(x, training=False)
        reconstruction       = self.decoder(z, training=False)
        total, recon, kl_unw, kl_w, reg = self._compute_losses(
            x, reconstruction, z_mean, z_log_var
        )

        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(recon)
        self.kl_unw_tracker.update_state(kl_unw)
        self.kl_w_tracker.update_state(kl_w)
        self.whiten_reg_tracker.update_state(reg)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss_unweighted": self.kl_unw_tracker.result(),
            "kl_loss": self.kl_w_tracker.result(),
            "whiten_reg": self.whiten_reg_tracker.result(),
        }


def asmsa_beta_vae(n_features, latent_dim=2, activation="gelu",
                   recon_loss_weight=1.0, beta=1e-4,
                   recon_fn=None):
    """
    Builder for the BetaVAE.
    Expects that `Sampling` and `asmsa_block` are already defined in scope.

    Args:
        n_features: total number of output features (e.g., nD + nA).
        latent_dim: latent dimensionality.
        activation: activation for MLP blocks.
        recon_loss_weight: global scale applied to reconstruction loss.
        beta: initial β value.
        recon_fn: callable(y_true, y_pred) -> per-sample loss (B,) or scalar.

    Returns:
        (vae, encoder, decoder)
    """
    # ----- Encoder -----
    enc_input = layers.Input(shape=(n_features,), name="enc_input")
    x = asmsa_block(enc_input, 128, activation, "enc1")
    x = asmsa_block(x,         64, activation, "enc2")
    x = asmsa_block(x,         32, activation, "enc3")

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling()([z_mean, z_log_var])
    encoder   = models.Model(enc_input, [z_mean, z_log_var, z], name="encoder")

    # (optional) init z_log_var bias for var < 1 at start
    zlv = encoder.get_layer("z_log_var")
    if hasattr(zlv, "bias") and zlv.bias is not None:
        try:
            zlv.bias.assign(tf.constant([-1.0] * latent_dim, dtype=tf.float32))
        except Exception:
            pass  # if not built yet, ignore

    # ----- Decoder -----
    dec_input  = layers.Input(shape=(latent_dim,), name="dec_input")
    y = asmsa_block(dec_input,  32, activation, "dec1")
    y = asmsa_block(y,          64, activation, "dec2")
    y = asmsa_block(y,         128, activation, "dec3")
    dec_output = layers.Dense(n_features, activation="linear", name="dec_output")(y)
    decoder    = models.Model(dec_input, dec_output, name="decoder")

    # ----- Optimizer -----
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999
    )

    # ----- Assemble -----
    vae = BetaVAE(
        encoder,
        decoder,
        recon_loss_weight=recon_loss_weight,
        beta=beta,          # initial β
        recon_fn=recon_fn,
        name="beta_vae"
    )
    vae.compile(optimizer=optimizer)
    return vae, encoder, decoder


    











