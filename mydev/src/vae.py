import tensorflow as tf
from tensorflow.keras import layers, models, metrics, initializers

# ==== Sampling e blocco denso ====
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.random.normal((batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def asmsa_block(x, neurons, activation, name_prefix, use_bn=True):
    x = layers.Dense(neurons, name=f"{name_prefix}_dense")(x)
    if use_bn:
        x = layers.BatchNormalization(momentum=0.8, name=f"{name_prefix}_normalization")(x)
    x = layers.Activation(activation, name=f"{name_prefix}_activation")(x)
    x = layers.Dropout(0.1, name=f"{name_prefix}_dropout")(x)
    return x


# ==== Modello Beta-VAE con training loop custom ====
class BetaVAE(models.Model):
    """
    Beta-VAE con:
      - β come tf.Variable non-trainabile (regolabile da callback)
      - Regolarizzatore opzionale di "whitening" (mean->0, cov->I) pesato da cov_reg
      - Training/test step custom
      - Metriche loggate: totale, ricostruzione, KL (unweighted e β-weighted),
        regolarizzatore, valore β corrente e peso cov_reg

    Puoi passare una loss di ricostruzione custom via `recon_fn(y_true, y_pred)`.
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

        # Variabili non-trainabili per controllo da callback
        self.beta    = tf.Variable(float(beta),    trainable=False, dtype=tf.float32, name="beta")
        self.cov_reg = tf.Variable(float(cov_reg), trainable=False, dtype=tf.float32, name="cov_reg")

        # Trackers / metriche
        self.total_loss_tracker          = metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_unw_tracker              = metrics.Mean(name="kl_loss_unweighted")  # nats per sample
        self.kl_w_tracker                = metrics.Mean(name="kl_loss")             # β * KL
        self.reg_tracker                 = metrics.Mean(name="whiten_reg")
        self.covw_tracker                = metrics.Mean(name="cov_reg_weight")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_unw_tracker,
            self.kl_w_tracker,
            self.reg_tracker,
            self.covw_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def _compute_losses(self, x, reconstruction, z_mean, z_log_var):
        # --- Reconstruction loss ---
        if self.recon_fn is not None:
            recon_raw = self.recon_fn(x, reconstruction)
            if isinstance(recon_raw, tf.Tensor) and recon_raw.shape.rank is not None and recon_raw.shape.rank >= 2:
                # riduci su tutte le dimensioni non-batch
                axes = list(range(1, recon_raw.shape.rank))
                recon_per_sample = tf.reduce_mean(recon_raw, axis=axes)
            else:
                # (B,) o scalare
                recon_per_sample = recon_raw
            recon_loss = tf.reduce_mean(recon_per_sample) * self.recon_loss_weight
        else:
            # Fallback: blend MSE/MAE per-sample
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

        # --- Whitening regularizer (sempre calcolato; scalato da cov_reg) ---
        mu = tf.cast(z_mean, tf.float32)                 # (B, d)
        b  = tf.cast(tf.shape(mu)[0], tf.float32)

        mean_mu  = tf.reduce_mean(mu, axis=0)           # (d,)
        mean_pen = tf.reduce_sum(tf.square(mean_mu))    # scalar

        mu_centered = mu - mean_mu
        denom = tf.maximum(b - 1.0, 1.0)                # evita div/0 quando batch=1
        cov = tf.matmul(mu_centered, mu_centered, transpose_a=True) / denom  # (d, d)
        eye = tf.eye(tf.shape(mu)[1], dtype=tf.float32)
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
        self.reg_tracker.update_state(reg)
        self.covw_tracker.update_state(self.cov_reg)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss_unweighted": self.kl_unw_tracker.result(),
            "kl_loss": self.kl_w_tracker.result(),
            "whiten_reg": self.reg_tracker.result(),
            "cov_reg_weight": self.covw_tracker.result(),
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
        self.reg_tracker.update_state(reg)
        self.covw_tracker.update_state(self.cov_reg)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss_unweighted": self.kl_unw_tracker.result(),
            "kl_loss": self.kl_w_tracker.result(),
            "whiten_reg": self.reg_tracker.result(),
            "cov_reg_weight": self.covw_tracker.result(),
        }


# ==== Builder dell'architettura ====
def asmsa_beta_vae(n_features, latent_dim=2, activation="gelu",
                   recon_loss_weight=1.0, beta=1e-4,
                   cov_reg=0.0,
                   recon_fn=None):
    """
    Costruttore del BetaVAE.

    Args:
        n_features: dimensione dell'input/output vettoriale.
        latent_dim: dimensione latente.
        activation: attivazione per i blocchi MLP.
        recon_loss_weight: scala globale per la loss di ricostruzione.
        beta: valore iniziale di β.
        cov_reg: peso del regolarizzatore di whitening.
        recon_fn: callable(y_true, y_pred) -> per-sample loss (B,) o scalare.

    Returns:
        (vae, encoder, decoder)
    """
    # ----- Encoder -----
    enc_input = layers.Input(shape=(n_features,), name="enc_input")
    x = asmsa_block(enc_input, 128, activation, "enc1", use_bn=False)
    x = asmsa_block(x,         64, activation, "enc2", use_bn=False)
    x = asmsa_block(x,         32, activation, "enc3", use_bn=False)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(
        latent_dim,
        name="z_log_var",
        bias_initializer=initializers.Constant(-1.0)  # var iniziale < 1
    )(x)
    z         = Sampling()([z_mean, z_log_var])
    encoder   = models.Model(enc_input, [z_mean, z_log_var, z], name="encoder")

    # ----- Decoder -----
    dec_input  = layers.Input(shape=(latent_dim,), name="dec_input")
    y = asmsa_block(dec_input,  32, activation, "dec1", use_bn=False)
    y = asmsa_block(y,          64, activation, "dec2", use_bn=False)
    y = asmsa_block(y,         128, activation, "dec3", use_bn=False)
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
        beta=beta,
        cov_reg=cov_reg,
        recon_fn=recon_fn,
        name="beta_vae"
    )
    vae.compile(optimizer=optimizer)
    return vae, encoder, decoder












