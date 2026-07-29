import numpy as np
from pathlib import Path
from typing import Dict, List

def check_degenerancy(file_path: str, tol: float = 1e-9) -> Dict[float, List[int]]:
    """
    Carga el diccionario exacto desde un archivo .npz y determina la dimensionalidad
    de la variedad degenerada del estado fundamental para cada valor de Jz.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    print("="*60)
    print(" ANÁLISIS DE DEGENERACIÓN DEL ESTADO FUNDAMENTAL")
    print("="*60)

    # Replicamos la lógica de carga de load_exact_results
    npz_file = np.load(path, allow_pickle=True)
    exact_results = npz_file['data_dict'].item()

    manifold_indices_dict = {}

    # Iteramos sobre el diccionario ordenando las claves (Jz)
    for jz_key in sorted(exact_results.keys()):
        data = exact_results[jz_key]

        # Extraemos el array completo de energías
        if 'energies' not in data:
            raise KeyError(f"Falta la clave 'energies' en Jz={jz_key}. Asegúrate de guardar todo el array.")

        energies = data['energies']
        E0_global = energies[0]

        # Aplicamos la máscara lógica para aislar los estados dentro del gap de tolerancia
        degenerate_mask = np.abs(energies - E0_global) < tol
        degenerate_indices = np.where(degenerate_mask)[0].tolist()
        manifold_dim = len(degenerate_indices)

        manifold_indices_dict[jz_key] = degenerate_indices

        print(f"\n[ Jz = {jz_key:.4f} ]")
        print(f"Energía E0 global: {E0_global:.8f}")
        print(f"Dimensionalidad del manifold: {manifold_dim}")

        for idx in degenerate_indices:
            delta_E = np.abs(energies[idx] - E0_global)
            print(f"  -> Estado {idx} | E = {energies[idx]:.8f} | ΔE = {delta_E:.2e}")

    print("\n" + "="*60)
    return manifold_indices_dict

if __name__ == "__main__":
    npz_path = 'data/raw/energies_eigenvecs_dict.npz'
    manifold_map = check_degenerancy(npz_path)