import tensorflow as tf

def split_dataset(features_normalized, train_size=70, val_size=15, batch_size=64, seed=42):
    """
    Crea dataset ottimizzati per training di autoencoder con validazioni e ottimizzazioni avanzate
    
    Args:
        features_normalized: array/tensor con features normalizzate
        train_size: percentuale per training (default 70%)
        val_size: percentuale per validation (default 15%)
        batch_size: dimensione dei batch (default 64)
        seed: seed per riproducibilità (default 42)
    
    Returns:
        tuple: (ds_train, ds_val, ds_test, ds_all, split_info)
    """
    import tensorflow as tf
    
    # Validazioni input
    if not isinstance(features_normalized, (tf.Tensor, tf.data.Dataset)):
        features_normalized = tf.convert_to_tensor(features_normalized)
    
    if len(features_normalized.shape) < 2:
        raise ValueError("features_normalized deve avere almeno 2 dimensioni")
    
    if train_size + val_size >= 100:
        raise ValueError(f"train_size ({train_size}) + val_size ({val_size}) deve essere < 100")
    
    if batch_size <= 0:
        raise ValueError("batch_size deve essere > 0")
    
    # Calcolo dimensioni
    n_samples = len(features_normalized)
    if n_samples < batch_size:
        print(f"Warning: n_samples ({n_samples}) < batch_size ({batch_size}). Ridotto batch_size a {n_samples}")
        batch_size = n_samples
    
    train_ratio = train_size / 100
    val_ratio = val_size / 100
    test_ratio = 1 - train_ratio - val_ratio
    
    n_train = max(1, int(train_ratio * n_samples))
    n_val = max(1, int(val_ratio * n_samples))
    n_test = n_samples - n_train - n_val
    
    # Assicurati che tutti i split abbiano almeno 1 campione
    if n_test < 1:
        n_test = 1
        n_train = n_samples - n_val - n_test
    
    # Dataset base con shuffle stratificato
    ds = tf.data.Dataset.from_tensor_slices(features_normalized)
    ds = ds.shuffle(buffer_size=n_samples, seed=seed, reshuffle_each_iteration=False)
    
    # Split dei dati
    ds_train = ds.take(n_train)
    ds_temp = ds.skip(n_train)
    ds_val = ds_temp.take(n_val)
    ds_test = ds_temp.skip(n_val)
    
    # Buffer size ottimizzato per training shuffle
    train_shuffle_buffer = min(1000, max(100, n_train // 10))
    
    # Pipeline di training ottimizzato
    ds_train = (ds_train
                .batch(batch_size, drop_remainder=True)
                .shuffle(buffer_size=train_shuffle_buffer, seed=seed)
                .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
                .cache())  # Cache per migliorare performance
    
    # Pipeline di validazione (senza shuffle per consistenza)
    ds_val = (ds_val
              .batch(batch_size, drop_remainder=False)
              .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE)
              .cache())
    
    # Pipeline di test (senza shuffle per riproducibilità)
    ds_test = (ds_test
               .batch(batch_size, drop_remainder=False)
               .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
               .prefetch(tf.data.AUTOTUNE)
               .cache())
    
    # Dataset completo per inference (mantenendo ordine originale)
    ds_all = (tf.data.Dataset.from_tensor_slices(features_normalized)
              .batch(batch_size, drop_remainder=False)
              .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))
    
    # Calcolo statistiche accurate
    try:
        train_batches = tf.data.experimental.cardinality(ds_train).numpy()
        val_batches = tf.data.experimental.cardinality(ds_val).numpy()
        test_batches = tf.data.experimental.cardinality(ds_test).numpy()
    except:
        # Fallback se cardinality() fallisce
        train_batches = (n_train + batch_size - 1) // batch_size  # ceil division
        val_batches = (n_val + batch_size - 1) // batch_size
        test_batches = (n_test + batch_size - 1) // batch_size
    
    # Informazioni dettagliate
    split_info = {
        'n_total': n_samples,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'train_ratio': n_train / n_samples,
        'val_ratio': n_val / n_samples,
        'test_ratio': n_test / n_samples,
        'batch_size': batch_size,
        'train_batches': train_batches,
        'val_batches': val_batches,
        'test_batches': test_batches,
        'seed': seed,
        'features_shape': features_normalized.shape
    }
    
    # Output dettagliato
    print(f"Dataset Split Statistics:")
    print(f"  Total samples: {n_samples}")
    print(f"  Train: {n_train} samples ({n_train/n_samples*100:.1f}%), {train_batches} batches")
    print(f"  Val:   {n_val} samples ({n_val/n_samples*100:.1f}%), {val_batches} batches") 
    print(f"  Test:  {n_test} samples ({n_test/n_samples*100:.1f}%), {test_batches} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Features shape: {features_normalized.shape}")
    print(f"  Seed: {seed}")
    
    return ds_train, ds_val, ds_test, ds_all, split_info


def validate_splits(ds_train, ds_val, ds_test, expected_features_shape):
    """
    Funzione di utilità per validare che gli split siano corretti
    """
    print("\nValidazione splits:")
    
    try:
        # Testa un batch da ogni dataset
        train_batch = next(iter(ds_train))
        val_batch = next(iter(ds_val))
        test_batch = next(iter(ds_test))
        
        print(f"  Train batch shape: {train_batch[0].shape}")
        print(f"  Val batch shape: {val_batch[0].shape}")
        print(f"  Test batch shape: {test_batch[0].shape}")
        
        # Verifica che input == output per autoencoder
        assert tf.reduce_all(train_batch[0] == train_batch[1]), "Train: input != output"
        assert tf.reduce_all(val_batch[0] == val_batch[1]), "Val: input != output"
        assert tf.reduce_all(test_batch[0] == test_batch[1]), "Test: input != output"
        
        print("  ✓ Tutti gli split sono validi per autoencoder")
        
    except Exception as e:
        print(f"  ✗ Errore nella validazione: {e}")
        return False
    
    return True

def asmsa_datasets(features, train_size=70, val_size=15, batch_size=64, seed=42):
    """
    Wrapper completo per creare dataset per autoencoder con gestione errori
    """
    try:
        # Crea gli split
        ds_train, ds_val, ds_test, ds_all, info = split_dataset(
            features, train_size, val_size, batch_size, seed
        )
        
        # Valida gli split
        is_valid = validate_splits(ds_train, ds_val, ds_test, features.shape)
        
        if not is_valid:
            raise ValueError("Validazione degli split fallita")
        
        return ds_train, ds_val, ds_test, ds_all, info
        
    except Exception as e:
        print(f"Errore nella creazione dataset: {e}")
        raise