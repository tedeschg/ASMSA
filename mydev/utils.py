import tensorflow as tf

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
