import numpy as np
import netket as nk
from typing import Tuple
from .hamiltonian import build_kitaev_lattice
# Asumiendo que has movido KitaevTransverse_H a hamiltonian.py
from .hamiltonian import KitaevTransverse_H 

def run_exact_diagonalization(extent = [3, 3], 
                              jz_steps: int = 11, 
                              k_eigenvals: int = 1,
                              save_path: str = 'data/raw/energies_eigenvecs.npz') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el estado fundamental exacto usando Lanczos sobre un rango de valores Jz.
    """
    graph, hilbert = build_kitaev_lattice(extent=extent, pbc=True)
    jz_values = np.linspace(0, 1, jz_steps)
    
    all_energies = []
    all_vecs = []

    for jz in jz_values:
        jx = jy = (1 - jz) / 2
        
        # OJO: Aquí requieres tu función KitaevTransverse_H. 
        # Asegúrate de importarla correctamente del nuevo archivo.
        H = KitaevTransverse_H(graph.edge_colors, graph.edges(), Jx=jx, Jy=jy, Jz=jz, h=0, hi=hilbert)
        
        eigenvals, eigenvecs = nk.exact.lanczos_ed(H, k=k_eigenvals, compute_eigenvectors=True)
        all_energies.append(eigenvals)
        all_vecs.append(eigenvecs)
        
        print(f"ED: Jz = {jz:.2f} finalizado")

    energies_array = np.array(all_energies)
    vecs_array = np.array(all_vecs)

    if save_path:
        np.savez(save_path, jz_values=jz_values, energies=energies_array, vecs=vecs_array)
        
    return jz_values, energies_array, vecs_array