# plot_observables.py
import os
import sys
import re
import glob
import pickle
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import netket as nk

from netket.operator.spin import sigmaz, sigmax, sigmay

# --- Importaciones de tu código limpio ---
from src.physics.hamiltonian import build_kitaev_lattice
from src.physics.symmetries import get_kitaev_symmetries
from src.physics.observables import get_kitaev_plaquettes, build_wilson_loops
from src.models.rbm import ProjectedRBM

# ============================================================
# 0. CONFIGURACIÓN Y EXPRESIONES REGULARES
# ============================================================
# TODO: Ajusta este patrón según cómo estés guardando los archivos en main.py
# Por defecto asume que el nombre contiene Layers y Jz. Ej: vstate_RBM_L2_Jz0.5.pkl
PKL_PATTERN = r"vstate_.*?L(\d+)_Jz([\d.]+).*\.pkl" 

def parse_pkl_name(fname):
    """Devuelve (layers, jz) o None si no encaja el patrón."""
    m = re.search(PKL_PATTERN, os.path.basename(fname))
    if m:
        return int(m.group(1)), float(m.group(2))
    return None

# ============================================================
# 1. HELPERS MATEMÁTICOS PARA OBSERVABLES EXTRAS
# ============================================================
def build_sparse_ops(hi, num_sites):
    ops = {'x': sigmax, 'y': sigmay, 'z': sigmaz}
    sparse_single_ops = {
        d: [ops[d](hi, i).to_sparse() for i in range(num_sites)]
        for d in ('x', 'y', 'z')
    }
    return sparse_single_ops

def magnetization_components(psi, sparse_single_ops, N_sites):
    M_vals, M_sq = {}, 0.0
    for d in ('x', 'y', 'z'):
        val = np.real(sum(np.vdot(psi, op @ psi) for op in sparse_single_ops[d])) / N_sites
        M_vals[d] = val
        M_sq += val ** 2
    M_vals['total'] = np.sqrt(M_sq)
    return M_vals

def S_corr_zz(hi, psi, sparse_ops_z):
    S = 0.0
    for i, opi in enumerate(sparse_ops_z):
        for j, opj in enumerate(sparse_ops_z):
            if i != j:
                S += np.real(np.vdot(psi, opi @ (opj @ psi)))
    return S

def quantum_coherence_l1(psi):
    abs_psi = np.abs(psi)
    return float(np.sum(abs_psi) ** 2 - np.sum(abs_psi ** 2))

def evaluate_wilson_loop(psi, Wp_total_op):
    """Evalúa el observable global de Wilson usando matrices dispersas para el estado denso."""
    # Convertimos el operador de NetKet a matriz dispersa para multiplicarlo por el vector psi
    Wp_sparse = Wp_total_op.to_sparse()
    return float(np.real(np.vdot(psi, Wp_sparse @ psi)))

# ============================================================
# MAIN SCRIPT
# ============================================================
def main():
    print("--- Inicializando Entorno para Análisis de Observables ---")
    
    # 1. Configurar Física (Igual que en run_vmc.py)
    extent = [3, 3]
    graph, hilbert = build_kitaev_lattice(extent=extent, pbc=True)
    N = graph.n_nodes
    
    plaquetas, ops_colores = get_kitaev_plaquettes(graph)
    Wp_list, Wp_total = build_wilson_loops(hilbert, plaquetas, ops_colores)
    
    symmetries_info = get_kitaev_symmetries(graph, hilbert)
    sparse_single_ops = build_sparse_ops(hilbert, N)
    sampler_dummy = nk.sampler.MetropolisLocal(hilbert)

    # Convertimos a Tuplas para que Flax no de el HashError
    irrep_index = 0
    symm_tuple = tuple(map(tuple, symmetries_info["irreps_matrices"][irrep_index].tolist()))
    char_tuple = tuple(symmetries_info["character_table"][irrep_index].tolist())

    # 2. Cargar Datos Exactos (Si existen)
    path_energies = 'data/raw/energies_eigenvecs.npz' # Ajusta la ruta a tu ED
    exact_dict = {}
    jz_values_exact = []
    
    if os.path.exists(path_energies):
        data_exact = np.load(path_energies)
        jz_values_exact = np.linspace(0, 1, 11)
        exact_dict = {
            round(jz_values_exact[i], 2): data_exact['vecs'][i, :, 0]
            for i in range(len(jz_values_exact))
        }
        print(f"[OK] Datos de Diagonalización Exacta cargados desde {path_energies}")
    else:
        print(f"[AVISO] No se encontró {path_energies}. Se omitirá el overlap.")

    # 3. Buscar PKLs en la carpeta local o de resultados
    SEARCH_DIRS = [".", "data/weights", "Resultados"]
    pkl_files = []
    
    for d in SEARCH_DIRS:
        if os.path.isdir(d):
            found = glob.glob(os.path.join(d, "*.pkl"))
            if found:
                pkl_files.extend(found)

    parsed = [(f, parse_pkl_name(f)) for f in pkl_files]
    valid = [(f, meta) for f, meta in parsed if meta is not None]

    if not valid:
        print("\n[ERROR] No se encontraron archivos .pkl válidos.")
        print(f"Asegúrate de que tus pkls cumplan el Regex: {PKL_PATTERN}")
        sys.exit(1)

    print(f"\n[OK] Encontrados {len(valid)} archivos .pkl válidos para analizar.")

    rows = []
    current_layers = None
    vstate = None

    # 4. Bucle de Evaluación
    for pkl_path, (layers, jz) in sorted(valid, key=lambda x: (x[1][0], x[1][1])):
        print(f"Evaluando: L={layers}, Jz={jz:.2f} -> {os.path.basename(pkl_path)}")

        # Solo reinstanciamos el modelo si cambia el número de capas
        if layers != current_layers:
            model = ProjectedRBM(
                num_layers=layers,
                alpha=1.0, 
                param_dtype=jnp.complex128,
                symmetries=symm_tuple,
                characters=char_tuple
            )
            vstate = nk.vqs.MCState(sampler_dummy, model, n_samples=2)
            current_layers = layers

        # Cargar pesos
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)
        vstate.parameters = params

        # Extraer vector de estado
        psi_rbm = vstate.to_array()
        psi_rbm = psi_rbm / np.linalg.norm(psi_rbm)

        # Calcular observables
        m = magnetization_components(psi_rbm, sparse_single_ops, N)
        Szz = S_corr_zz(hilbert, psi_rbm, sparse_single_ops['z'])
        Cl1 = quantum_coherence_l1(psi_rbm)
        Wp_val = evaluate_wilson_loop(psi_rbm, Wp_total)

        # Overlap
        jz_key = round(jz, 2)
        overlap = float('nan')
        if jz_key in exact_dict:
            psi_exact = exact_dict[jz_key] / np.linalg.norm(exact_dict[jz_key])
            overlap = float(np.abs(np.vdot(psi_exact, psi_rbm)) ** 2)

        rows.append({
            'pkl': os.path.basename(pkl_path),
            'layers': layers,
            'Jz': jz,
            'Mx': m['x'], 'My': m['y'], 'Mz': m['z'], 'M_total': m['total'],
            'Szz': Szz,
            'Cl1': Cl1,
            'Wp': Wp_val,
            'overlap': overlap
        })

        del psi_rbm, params
        gc.collect()

    # 5. Guardar y Graficar
    df = pd.DataFrame(rows)
    df.to_csv('resultados_observables.csv', index=False)
    print("\n[OK] CSV guardado: resultados_observables.csv")

    if df.empty: return

    # Pre-calcular Exactos para los Plots
    obs_exact = {'Mx': [], 'My': [], 'Mz': [], 'M_total': [], 'Szz': [], 'Cl1': [], 'Wp': []}
    if jz_values_exact:
        Wp_sparse = Wp_total.to_sparse()
        for i, jz_i in enumerate(jz_values_exact):
            psi = data_exact['vecs'][i, :, 0]
            psi = psi / np.linalg.norm(psi)
            m = magnetization_components(psi, sparse_single_ops, N)
            
            obs_exact['Mx'].append(m['x'])
            obs_exact['My'].append(m['y'])
            obs_exact['Mz'].append(m['z'])
            obs_exact['M_total'].append(m['total'])
            obs_exact['Szz'].append(S_corr_zz(hilbert, psi, sparse_single_ops['z']))
            obs_exact['Cl1'].append(quantum_coherence_l1(psi))
            obs_exact['Wp'].append(float(np.real(np.vdot(psi, Wp_sparse @ psi))))

    # Configuración de Gráficas
    observables_config = [
        ('M_total', obs_exact.get('M_total', None), '$M_{total}$', 'tab:orange'),
        ('Mz',      obs_exact.get('Mz', None),      '$M_z$',       'tab:red'),
        ('Szz',     obs_exact.get('Szz', None),     '$S^{zz}$',    'tab:purple'),
        ('Cl1',     obs_exact.get('Cl1', None),     '$C_{l1}$',    'tab:cyan'),
        ('Wp',      obs_exact.get('Wp', None),      'Wilson Loop $\\langle W_p \\rangle$', 'tab:green'),
        ('overlap', None, 'Overlap $|\\langle\\psi_{ED}|\\psi_{RBM}\\rangle|^2$', 'crimson'),
    ]

    layer_vals = sorted(df['layers'].unique())
    cmap = plt.cm.tab10

    os.makedirs("plots", exist_ok=True)

    for obs_col, exact_vals, ylabel, exact_color in observables_config:
        fig, ax = plt.subplots(figsize=(8, 5))

        if exact_vals is not None and len(exact_vals) > 0:
            ax.plot(jz_values_exact, exact_vals, 'o-', color=exact_color, linewidth=2, label='Exacto (ED)', zorder=2)

        for li, layer in enumerate(layer_vals):
            sub = df[df['layers'] == layer].sort_values('Jz')
            if obs_col not in sub.columns or sub[obs_col].isna().all():
                continue
            
            ax.plot(sub['Jz'], sub[obs_col], marker='s', markersize=8, color=cmap(li), label=f'RBM layers={layer}', zorder=5, linestyle='--')

        ax.set_xlabel('$J_z$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f'{ylabel} vs $J_z$', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        fname = f'plots/obs_{obs_col}_vs_jz.png'
        plt.savefig(fname, dpi=300)
        plt.close()
        
    print("\n[OK] ¡Todos los plots guardados en la carpeta 'plots/'!")

if __name__ == "__main__":
    main()