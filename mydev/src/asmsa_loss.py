import tensorflow as tf

def asmsa_ae_loss(nD, nA, *, deltaD=0.1, deltaA=0.5, wD=None, wA=None):
    """
    Reconstruction loss for autoencoder with features = [distances || angle_sin_cos].
    - Distances: Huber(deltaD)
    - Angles (sin/cos): Huber(deltaA) directly on sin/cos (no wrapping)
    - Weights: if not provided, weights are based on feature counts (nD, nA).
               Otherwise (wD, wA) are normalized.

    Args:
        nD: number of distance features
        nA: number of angular features (sum of all sine and cosine columns)
        deltaD: Huber delta for distances
        deltaA: Huber delta for sin/cos
        wD, wA: optional weights for (distances, angles)
    """
    huberD = tf.keras.losses.Huber(delta=deltaD, reduction=tf.keras.losses.Reduction.NONE)
    huberA = tf.keras.losses.Huber(delta=deltaA, reduction=tf.keras.losses.Reduction.NONE)

    # normalized weights
    if wD is None or wA is None:
        wD_t = nD / float(nD + nA) if (nD + nA) > 0 else 0.0
        wA_t = nA / float(nD + nA) if (nD + nA) > 0 else 0.0
    else:
        s = float(wD + wA)
        wD_t, wA_t = (wD / s, wA / s) if s > 0 else (0.5, 0.5)

    def loss(y_true, y_pred):
        d_true = y_true[:, :nD]
        d_pred = y_pred[:, :nD]
        a_true = y_true[:, nD:nD + nA]
        a_pred = y_pred[:, nD:nD + nA]

        # Huber per-feature → mean over features → per-sample value
        lossD = tf.reduce_mean(huberD(d_true, d_pred), axis=-1)
        lossA = tf.reduce_mean(huberA(a_true, a_pred), axis=-1)

        return wD_t * lossD + wA_t * lossA

    return loss
