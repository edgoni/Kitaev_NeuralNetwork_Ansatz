import json
from pathlib import Path
import numpy as np
import netket as nk
from typing import Dict, Any
from .hamiltonian import build_kitaev_lattice, KitaevTransverse_H 

def _convert_for_json(obj: Any) -> Any:
    """
    Convierte tipos nativos de NumPy a tipos estándar de Python para JSON.
    """
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    elif isinstance(obj, dict):
        return {str(k): _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_for_json(x) for x in obj]
    return obj

def run_exact_diagonalization(
    extent: list = [3, 3], 
    jz_steps: int = 11, 
    k_eigenvals: int = 1,
    save_path: str = 'data/raw/energies_eigenvecs_dict.npz',
    save_debug_json: bool = True
) -> Dict[float, Dict[str, Any]]:
    
    graph, hilbert = build_kitaev_lattice(extent=extent, pbc=True)
    jz_values = np.linspace(0, 1, jz_steps)
    
    exact_results = {}
    json_debug_results = {}

    for jz in jz_values:
        jx = jy = (1.0 - jz) / 2.0
        H = KitaevTransverse_H(graph.edge_colors, graph.edges(), Jx=jx, Jy=jy, Jz=jz, h=0, hi=hilbert)
        eigenvals, eigenvecs = nk.exact.lanczos_ed(H, k=k_eigenvals, compute_eigenvectors=True)
        
        irrep_contributions = identify_irreps(eigenvecs[:, 0], hilbert, graph.automorphisms(), graph.space_group().character_table())
        # Clave numérica redondeada para evitar errores de flotantes
        jz_key = round(float(jz), 4)
        exact_results[jz_key] = {
            'E0': float(eigenvals[0].real),
            'psi0': eigenvecs[:, 0],
            'irrep_contributions': irrep_contributions
        }

        json_debug_results[jz_key] = {
            'E0': float(eigenvals[0].real),
            'irrep_contributions': irrep_contributions
        }

        print(f"[ED] Jz = {jz_key:.4f} completado -> E0 = {float(eigenvals[0].real):.6f}")

    if save_path:
        # Volcado binario comprimido (.npz)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, data_dict=exact_results)
        print(f"[SUCCESS] Diccionario binario completo guardado en: {save_path}")
        
        # Volcado JSON ligero (sin vectores de estado)
        if save_debug_json:
            json_path = Path(save_path).with_suffix('.json')
            clean_json_dict = _convert_for_json(json_debug_results)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(clean_json_dict, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] Archivo JSON (solo autovalores e irreps) guardado en: {json_path}")

    return exact_results

    if save_path:
        # Guardamos el diccionario completo empaquetado bajo la clave 'data_dict'
        np.savez_compressed(save_path, data_dict=exact_results)
        print(f"[SUCCESS] Diccionario guardado en formato .npz: {save_path}")
        np.save
    return exact_results


def load_exact_results(file_path: str = 'data/raw/energies_eigenvecs_dict.npz') -> Dict[float, Dict[str, Any]]:
    """
    Carga el diccionario exacto desde un archivo .npz.
    Requiere allow_pickle=True y extraer el objeto escalar con .item()
    """
    npz_file = np.load(file_path, allow_pickle=True)
    exact_results = npz_file['data_dict'].item()
    return exact_results


def identify_irreps(eigvec, hi, sg, character_table) -> Dict[int, float]:
    '''
    Function to identify the irreps of the eigenstates of a Hamiltonian,
    given the symmetry group and its character table.
    :param eigvec: Eigenvector of the Hamiltonian
    :param hi: NetKet Hilbert space
    :param sg: Symmetry group of the system
    :param character_table: Character table of the symmetry group
    '''

    n_g = len(sg)
    n_irreps = character_table.shape[0]
        
    expect_vals = []
    for g in sg:
        op = nk.operator.permutation.PermutationOperator(hi, g)
        val = np.vdot(eigvec, op.to_sparse() @ eigvec)
        expect_vals.append(val)
        
    expect_vals = np.array(expect_vals)
    pesos = []

    for i in range(n_irreps):
        d_mu = np.real(character_table[i, 0])
        peso = (d_mu / n_g) * np.sum(np.conj(character_table[i, :]) * expect_vals)
        pesos.append(peso.real)

    irrep_contribution = {i: peso for i, peso in enumerate(pesos)}

    return irrep_contribution

