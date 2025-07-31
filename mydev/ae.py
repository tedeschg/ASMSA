import tensorflow as tf
from tensorflow.keras import layers, models, Model,regularizers

def asmsa_ae(n_features, latent_dim=2, activation="gelu"):

    def asmsa_block(x, neurons, name_prefix):
        x = layers.Dense(neurons, name=f"{name_prefix}_dense" )(x)
        x = layers.BatchNormalization(momentum=0.8, name=f"{name_prefix}_normalization")(x)
        x = layers.Activation(activation, name=f"{name_prefix}_activation")(x)
        x = layers.Dropout(0.1, name=f"{name_prefix}_dropout" )(x)
        return x

    enc_input_layer = layers.Input(shape=(n_features,), name="enc_input")
    
    x = asmsa_block(enc_input_layer, 128, "enc_1")
    x = asmsa_block(x, 64, "enc_2")
    x = asmsa_block(x, 32, "enc_3")


    latent = layers.Dense(latent_dim, name="latent")(x)
    encoder = models.Model(inputs=enc_input_layer, outputs=latent, name="encoder")

    dec_input_layer = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = asmsa_block(dec_input_layer, 128, "dec_1")
    x = asmsa_block(x, 64, "dec_2")
    x = asmsa_block(x, 32, "dec_3")
    dec_output_layer = layers.Dense(n_features, activation="linear", name="dec_output_layer")(x)
    decoder = models.Model(inputs=dec_input_layer, outputs=dec_output_layer, name="decoder")

    # --- Autoencoder ---
    autoencoder = models.Model(inputs=enc_input_layer, outputs=decoder(encoder(enc_input_layer)), name="autoencoder")

    return autoencoder, encoder, decoder

def compile_asmsa_ae(autoencoder, learning_rate=1e-4):

    # AdamW works with GELU
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-5, 
        beta_1=0.9,
        beta_2=0.999
    )
    
    autoencoder.compile(
        optimizer=optimizer,
        loss='mse'
    )
    
    return autoencoder