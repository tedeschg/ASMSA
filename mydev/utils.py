import tensorflow as tf
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

import numpy as np
import mdtraj as md
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, metrics, backend as K

def process_trajectory(traj, conf):
    
    traj = md.load_xtc(traj, top=conf)
    backbone_atoms = traj.topology.select('backbone')
    traj.superpose(traj, 0, atom_indices=backbone_atoms)

    # numero di frame
    n_frames, n_atoms = traj.n_frames, traj.n_atoms

    # selezione atomi
    p_indices = traj.topology.select("protein")
    n_p = len(p_indices)

    bb_indices = traj.topology.select("backbone")
    n_bb = len(bb_indices)

    ca_indices = traj.topology.select("name CA")
    n_ca = len(ca_indices)
    # tutte le coppie i<j di CA
    pairs = np.array([
        (i, j) 
        for idx, i in enumerate(ca_indices) 
        for j in ca_indices[idx+1:]
    ])

    # coordinate backbone [n_frames, n_bb, 3]
    coords_bb = traj.xyz[:, bb_indices, :]
    # [n_frames, n_bb*3]
    coords = coords_bb.reshape(n_frames, -1)

    # angoli di backbone (phi, psi)
    phi = md.compute_phi(traj)[1]
    psi = md.compute_psi(traj)[1]
    phi_sin = np.sin(phi)
    phi_cos = np.cos(phi)
    psi_sin = np.sin(psi)
    psi_cos = np.cos(psi)

    # angoli di catena laterale (chi1, chi2), se presenti
    chi1 = md.compute_chi1(traj)[1]
    chi2 = md.compute_chi2(traj)[1]
    chi1_sin = np.sin(chi1)
    chi1_cos = np.cos(chi1)
    chi2_sin = np.sin(chi2)
    chi2_cos = np.cos(chi2)

    # concatenazione di tutte le feature
    feat = np.concatenate([
        coords,
        phi_sin, phi_cos,
        psi_sin, psi_cos
    ], axis=1)
 
    # normalizzazione [0,1]
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(feat)

    return ca_indices, n_ca, bb_indices, n_bb, features_normalized, scaler, coords


def split_dataset(features_normalized, train_size=70, val_size=15, batch_size=64, seed=42):
    """
    Crea dataset ottimizzati per training di autoencoder
    """
    n_samples = len(features_normalized)

    train_size = train_size / 100
    val_size = val_size / 100

    n_train = int(train_size * n_samples)
    n_val = int(val_size * n_samples)
    n_test = n_samples - n_train - n_val
    
    # Dataset base con shuffle iniziale
    ds = tf.data.Dataset.from_tensor_slices(features_normalized)
    ds = ds.shuffle(buffer_size=n_samples, seed=seed, reshuffle_each_iteration=False)
    
    # Split dei dati
    ds_train = ds.take(n_train)
    ds_temp = ds.skip(n_train)
    ds_val = ds_temp.take(n_val)
    ds_test = ds_temp.skip(n_val)
    
    # Pipeline di training ottimizzato
    ds_train = ds_train.batch(batch_size, drop_remainder=True) \
                       .shuffle(buffer_size=max(100, n_train // batch_size // 10)) \
                       .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE) \
                       .prefetch(tf.data.AUTOTUNE)
    
    # Pipeline di validazione
    ds_val = ds_val.batch(batch_size, drop_remainder=False) \
                   .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE) \
                   .prefetch(tf.data.AUTOTUNE)
    
    # Pipeline di test
    ds_test = ds_test.batch(batch_size, drop_remainder=False) \
                     .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE) \
                     .prefetch(tf.data.AUTOTUNE)
    
    # Dataset completo per inference
    ds_all = tf.data.Dataset.from_tensor_slices(features_normalized) \
                           .batch(batch_size, drop_remainder=False) \
                           .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE) \
                           .prefetch(tf.data.AUTOTUNE)
    
    # Statistiche
    train_batches = tf.data.experimental.cardinality(ds_train).numpy()
    val_batches = tf.data.experimental.cardinality(ds_val).numpy()
    test_batches = tf.data.experimental.cardinality(ds_test).numpy()
    
    print(f"Dataset Statistics:")
    print(f"  Train: {n_train} samples, {train_batches} batches")
    print(f"  Val:   {n_val} samples, {val_batches} batches") 
    print(f"  Test:  {n_test} samples, {test_batches} batches")
    print(f"  Batch size: {batch_size}")
    
    return ds_train, ds_val, ds_test, ds_all

def plot_latent_space(latent_dim, encoder, dataset, conf, traj, target, bb_indices, cmap='rainbow', figsize=(8,8), model="ae", exact=True):

    # Get embeddings
    results = encoder.predict(dataset)

    if model == "ae":
        emb = np.array(results)
    elif model == "vae":
        emb = np.array(results[2])  #  [2] z from (z_mean, z_log_var, z)
    else:
        raise ValueError(f"Unknown model type: {model}. Use 'ae' or 'vae'.") 

    rms_ref = md.load_pdb(conf)
    rms_ref_bb   = rms_ref.atom_slice(bb_indices)
    rms_tr = md.load_xtc(traj, top=rms_ref)
    rmsd = md.rmsd(rms_tr, rms_ref)

    if model == 'vae':
        z = np.random.normal(loc=0.0, scale=1.0, size=(latent_dim,))
        sample = z
    elif model == 'ae':
        if exact == True:
            dists = np.linalg.norm(emb - target, axis=1)
            idx_closest = np.argsort(dists)[1]
            sample = emb[idx_closest].reshape(1, latent_dim)
        elif exact == False:
            sample = target
            
    plt.figure(figsize=figsize)

    plt.scatter(emb[:,0], emb[:,1], c=rmsd,s=0.5, cmap=cmap)
    plt.scatter(sample[:,0], sample[:,1], marker="X", c="Black")

    plt.show()
    return emb, sample



# === Callback semplice per monitoraggio ===
class BetaVAEMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            kl_loss = logs.get('kl_loss', 0)
            recon_loss = logs.get('reconstruction_loss', 0)
            print(f"\nEpoca {epoch+1}: Beta={self.model.beta:.4f}, "
                  f"KL={kl_loss:.6f}, Recon={recon_loss:.6f}")

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
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'ae_{latent_dim}d.keras',
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )]
    if model == 'vae':
        cb.append(BetaVAEMonitor())


    return cb
'''
class ProteinStructureMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            var_ratio = logs.get('val_var_ratio_metric', 0)
            if var_ratio > 1.5 or var_ratio < 0.5:
                print(f"⚠️  Epoch {epoch}: Variance ratio anomalo: {var_ratio:.4f}")
                '''