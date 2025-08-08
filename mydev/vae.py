import tensorflow as tf
from tensorflow.keras import layers, models, metrics

# === Sampling and blocks ===
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal((batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def asmsa_block(x, neurons, activation, name_prefix):
    x = layers.Dense(neurons, name=f"{name_prefix}_dense")(x)
    x = layers.BatchNormalization(momentum=0.8, name=f"{name_prefix}_bn")(x)
    x = layers.Activation(activation, name=f"{name_prefix}_act")(x)
    x = layers.Dropout(0.1, name=f"{name_prefix}_dropout")(x)
    return x

# === Custom VAE and Monitor ===
class BetaVAE(models.Model):
    def __init__(self, encoder, decoder, recon_loss_weight=1.0, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_weight = recon_loss_weight
        self.beta = beta
        self.total_loss_tracker = metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            mse = tf.reduce_mean(tf.square(x - reconstruction))
            mae = tf.reduce_mean(tf.abs(x - reconstruction))
            recon_loss = (0.7 * mse + 0.3 * mae) * self.recon_loss_weight

            z_log_var_clamped = tf.clip_by_value(z_log_var, -10.0, 10.0)
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var_clamped
                              - tf.square(z_mean)
                              - tf.exp(z_log_var_clamped),
                              axis=1)
            )
            kl_loss = kl * self.beta
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)

        mse = tf.reduce_mean(tf.square(x - reconstruction))
        mae = tf.reduce_mean(tf.abs(x - reconstruction))
        recon_loss = (0.7 * mse + 0.3 * mae) * self.recon_loss_weight

        z_log_var_clamped = tf.clip_by_value(z_log_var, -10.0, 10.0)
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var_clamped
                          - tf.square(z_mean)
                          - tf.exp(z_log_var_clamped),
                          axis=1)
        )
        kl_loss = kl * self.beta
        total_loss = recon_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class BetaVAEMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            kl_loss = logs.get('kl_loss', 0)
            recon_loss = logs.get('reconstruction_loss', 0)
            print(f"\nEpoca {epoch+1}: Beta={self.model.beta:.4f}, "
                  f"KL={kl_loss:.6f}, Recon={recon_loss:.6f}")

# === Builder Function ===
def asmsa_beta_vae(n_features, latent_dim=2,
                         activation="gelu",
                         recon_loss_weight=1.0, beta=0.1):
    # Encoder
    enc_input = layers.Input(shape=(n_features,), name="enc_input")
    x = asmsa_block(enc_input, 128, activation, "enc1")
    x = asmsa_block(x, 64, activation, "enc2")
    x = asmsa_block(x, 32, activation, "enc3")

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(enc_input, [z_mean, z_log_var, z], name="encoder")

    # Initialize log_var bias for more variance
    for layer in encoder.layers:
        if 'z_log_var' in getattr(layer, 'name', '') and hasattr(layer, 'bias'):
            layer.bias.assign(tf.constant([-1.0] * layer.units, dtype=tf.float32))

    # Decoder
    dec_input = layers.Input(shape=(latent_dim,), name="dec_input")
    y = asmsa_block(dec_input, 32, activation, "dec1")
    y = asmsa_block(y, 64, activation, "dec2")
    y = asmsa_block(y, 128, activation, "dec3")
    dec_output = layers.Dense(n_features, activation="sigmoid", name="dec_output")(y)
    decoder = models.Model(dec_input, dec_output, name="decoder")

    # Assemble and compile VAE
    vae = BetaVAE(encoder, decoder,
                  recon_loss_weight=recon_loss_weight,
                  beta=beta,
                  name="beta_vae")
    vae.compile(
        optimizer=tf.keras.optimizers.Adam(),
    )

    return vae, encoder, decoder











