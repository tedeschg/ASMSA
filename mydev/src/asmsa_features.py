import numpy as np
import mdtraj as md
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def process_trajectory(traj, conf):
    traj = md.load_xtc(traj, top=conf)
    backbone_atoms = traj.topology.select('backbone')
    traj.superpose(traj, 0, atom_indices=backbone_atoms)
    
    # Numero di frame
    n_frames, n_atoms = traj.n_frames, traj.n_atoms
    
    # Selezione atomi
    p_indices = traj.topology.select("protein")
    n_p = len(p_indices)
    bb_indices = traj.topology.select("backbone")
    n_bb = len(bb_indices)
    ca_indices = traj.topology.select("name CA")
    n_ca = len(ca_indices)
    
    # Coordinate protein [n_frames, n_p, 3]
    coords_p = traj.xyz[:, p_indices, :]
    coords = coords_p.reshape(n_frames, -1)
    
    # Angoli di backbone (phi, psi)
    phi = md.compute_phi(traj)[1]
    psi = md.compute_psi(traj)[1]
    
    # Angoli di catena laterale (chi1, chi2), se presenti
    chi1 = md.compute_chi1(traj)[1]
    chi2 = md.compute_chi2(traj)[1]
    
    # APPROCCIO 1: Rappresentazione sin/cos (RACCOMANDATO)
    # Converte automaticamente gli angoli in spazio continuo
    phi_sin, phi_cos = np.sin(phi), np.cos(phi)
    psi_sin, psi_cos = np.sin(psi), np.cos(psi)
    chi1_sin, chi1_cos = np.sin(chi1), np.cos(chi1)
    chi2_sin, chi2_cos = np.sin(chi2), np.cos(chi2)
    
    # Concatena tutte le features angolari
    angle_features = np.concatenate([
        phi_sin, phi_cos, psi_sin, psi_cos,
        chi1_sin, chi1_cos, chi2_sin, chi2_cos
    ], axis=1)
    
    # NORMALIZZAZIONE SEPARATA
    # Coordinate: MinMax scaling
    scaler_coords = MinMaxScaler(feature_range=(-1, 1))
    coords_normalized = scaler_coords.fit_transform(coords)
    
    # Angoli (sin/cos): già in [-1,1], ma possiamo standardizzare
    scaler_angles = MinMaxScaler(feature_range=(-1, 1))  # O MinMaxScaler se preferisci
    angles_normalized = scaler_angles.fit_transform(angle_features)
    
    # Combina features normalizzate
    features_normalized = np.concatenate([
        coords_normalized,
        angles_normalized
    ], axis=1)
    
    # Restituisci anche i limiti per separare coord da angoli nel decoding
    n_coord_features = coords_normalized.shape[1]
    n_angle_features = angles_normalized.shape[1]
    
    return {
        'ca_indices': ca_indices,
        'n_ca': n_ca,
        'bb_indices': bb_indices,
        'n_bb': n_bb,
        'features_normalized': features_normalized,
        'scaler_coords': scaler_coords,
        'scaler_angles': scaler_angles,
        'coords': coords,
        'n_coord_features': n_coord_features,
        'n_angle_features': n_angle_features,
        'raw_angles': {
            'phi': phi, 'psi': psi, 'chi1': chi1, 'chi2': chi2
        }
    }