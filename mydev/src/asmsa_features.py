import numpy as np
import mdtraj as md
from sklearn.preprocessing import MinMaxScaler

# ---------- helpers per coppie NB ----------
def _dense_pairs(n_local):
    # tutte le coppie i<j in [0, n_local)
    I, J = np.triu_indices(n_local, k=1)
    return np.stack([I, J], axis=1)

def _sparse_pairs(n_local, density=1):
    """
    Grafo tipo ring-lattice: per ogni "step" 1..density collega i con (i+step)%n.
    È vicino alla logica del tuo chordal-cycle e scala O(n * density).
    """
    pairs = set()
    for step in range(1, density + 1):
        for a in range(n_local):
            b = (a + step) % n_local
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    pairs = np.array(sorted(pairs))
    return pairs

def _select_atoms(traj, atom_selection):
    """
    atom_selection: 'protein' | 'backbone' | 'CA' | list/np.array di indici globali
    """
    if isinstance(atom_selection, (list, np.ndarray)):
        atoms = np.array(atom_selection, dtype=int)
    elif atom_selection == "protein":
        atoms = traj.topology.select("protein")
    elif atom_selection == "backbone":
        atoms = traj.topology.select("backbone")
    elif atom_selection in ("CA", "alpha", "alphac", "Cα"):
        atoms = traj.topology.select("name CA")
    else:
        # default: proteina
        atoms = traj.topology.select("protein")
    return np.asarray(atoms, dtype=int)

# ---------- funzione principale ----------
def process_trajectory(
    traj, conf,
    atom_selection="CA",
    distance_mode="sparse",      # 'sparse' | 'dense'
    density=2,                   # usato se sparse
    include_angles=True,         # se False ritorna solo distanze
    superpose_on="backbone"      # None | 'backbone' | 'CA'
):
    """
    Sostituisce le coordinate con un vettore di distanze NB (sparse o dense).
    Normalizza distanze in [-1,1]; angoli in [-1,1] (sin/cos già in range, ma ri-scala per uniformità).
    """

    traj = md.load_xtc(traj, top=conf)

    # superpose opzionale
    if superpose_on is not None:
        if superpose_on == "backbone":
            atom_ids = traj.topology.select('backbone')
        elif superpose_on in ("CA", "alpha", "alphac", "Cα"):
            atom_ids = traj.topology.select('name CA')
        else:
            atom_ids = traj.topology.select('backbone')
        if atom_ids.size > 0:
            traj.superpose(traj, 0, atom_indices=atom_ids)

    n_frames, n_atoms = traj.n_frames, traj.n_atoms

    # --- metadati utili ---
    p_indices  = traj.topology.select("protein")
    bb_indices = traj.topology.select("backbone")
    ca_indices = traj.topology.select("name CA")

    n_p  = len(p_indices)
    n_bb = len(bb_indices)
    n_ca = len(ca_indices)

    # --- selezione atomi su cui costruire le distanze ---
    atoms_sel = _select_atoms(traj, atom_selection)
    n_sel = len(atoms_sel)
    if n_sel < 2:
        raise ValueError("Servono almeno 2 atomi per calcolare distanze.")

    # --- costruzione coppie (in indice locale) ---
    if distance_mode == "dense":
        local_pairs = _dense_pairs(n_sel)
    elif distance_mode == "sparse":
        local_pairs = _sparse_pairs(n_sel, density=max(1, int(density)))
    else:
        raise ValueError("distance_mode deve essere 'sparse' o 'dense'.")

    # mappa a indici globali del topology
    global_pairs = atoms_sel[local_pairs]  # shape (n_pairs, 2)

    # --- distanze NB per tutti i frame ---
    # Usa MDTraj per avere PBC corrette se presenti nella topologia
    dists = md.compute_distances(traj, global_pairs)  # (n_frames, n_pairs)

    # --- normalizzazione distanze ---
    scaler_dists = MinMaxScaler(feature_range=(-1, 1))
    dists_normalized = scaler_dists.fit_transform(dists)

    # --- angoli (opzionale) ---
    angle_features = np.empty((n_frames, 0), dtype=np.float32)
    scaler_angles = None
    raw_angles = {}

    if include_angles:
        # phi, psi
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)

        # chi1/chi2 (possono essere vuoti)
        try:
            _, chi1 = md.compute_chi1(traj)
        except Exception:
            chi1 = np.empty((n_frames, 0))
        try:
            _, chi2 = md.compute_chi2(traj)
        except Exception:
            chi2 = np.empty((n_frames, 0))

        # rappresentazione sin/cos
        def _sin_cos(a):
            if a.size == 0:
                return np.empty((n_frames, 0)), np.empty((n_frames, 0))
            return np.sin(a), np.cos(a)

        phi_sin, phi_cos = _sin_cos(phi)
        psi_sin, psi_cos = _sin_cos(psi)
        #chi1_sin, chi1_cos = _sin_cos(chi1)
        #chi2_sin, chi2_cos = _sin_cos(chi2)

        angle_features = np.concatenate(
            [phi_sin, phi_cos, psi_sin, psi_cos], #chi1_sin, chi1_cos, chi2_sin, chi2_cos],
            axis=1
        )

        scaler_angles = MinMaxScaler(feature_range=(-1, 1))
        # anche se sono già in [-1,1], ri-scaliamo per coerenza numerica
        angles_normalized = scaler_angles.fit_transform(angle_features)
    else:
        angles_normalized = np.empty((n_frames, 0), dtype=np.float32)

    # --- feature finali ---
    features_normalized = np.concatenate([dists_normalized, angle_features], axis=1)

    return {
        # indici utili
        'ca_indices': ca_indices,
        'n_ca': n_ca,
        'bb_indices': bb_indices,
        'n_bb': n_bb,
        'p_indices': p_indices,
        'n_p': n_p,
        'atom_indices_used': atoms_sel,

        # distanze
        'distance_pairs': global_pairs,       # shape (n_pairs, 2), indici globali
        'n_pairs': global_pairs.shape[0],
        'dists': dists,                       # raw (nm)
        'scaler_dists': scaler_dists,
        'n_distance_features': dists.shape[1],

        # angoli
        'raw_angles': {'phi': phi if include_angles else None,
                       'psi': psi if include_angles else None,
                       #'chi1': chi1 if include_angles else None,
                       #'chi2': chi2 if include_angles else None
                       },
        'scaler_angles': scaler_angles,
        'n_angle_features': angles_normalized.shape[1],

        # feature combinate
        'features_normalized': features_normalized,
    }
