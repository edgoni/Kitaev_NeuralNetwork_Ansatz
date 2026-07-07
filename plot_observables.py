#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
TFM: Neural-Network Quantum States for Many-Body Quantum Spin Systems
Author: Eduardo Goñi Crespo
Advisor: Master's Thesis Advisor & Computational Physics Expert

Module: evaluate_all_observables.py
Description:
    Evaluación exhaustiva de estados variacionales NQS (.mpack) frente a 
    Diagonalización Exacta (ED). Calcula estimadores MCMC y densos exactos:
      - Energía del Estado Fundamental (E0 / N)
      - Componentes de Magnetización (<Mx>, <My>, <Mz>, M_total)
      - Correlación de espín a dos cuerpos (<Szz>)
      - Operadores de bucle de Wilson plaquetarios (<Wp>)
      - Coherencia Cuántica en norma l1 (C_l1)
      - Fidelidad Cuántica Exacta / Overlap (|⟨ψ_ED | ψ_NQS⟩|²)
=============================================================================
"""

import os
import re
import glob
import gc
import pickle
import warnings
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax.serialization
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz

# Importaciones del repositorio (física y modelos)
try:
    from src.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
    from src.physics.exact_diag import run_exact_diagonalization, load_exact_results
    from src.models.rbm import RBM
    from src.models.quantumself import QuantumSelfAttention
    from src.physics.observables import build_sparse_observables, calculate_dense_metrics
except ImportError:
    warnings.warn(
        "[WARN] Módulos de 'src/' no encontrados mediante importación absoluta. "
        "Asegúrate de ejecutar desde la raíz del repositorio."
    )

# Estilo académico para gráficos LaTeX-ready
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
})


# =============================================================================
# 3. METADATA PARSING & HELPERS
# =============================================================================
MPACK_PATTERN = re.compile(r"^([A-Za-z]+)_.*N?(\d+).*_Jz([0-9.]+)\.mpack$")

def parse_mpack_filename(filepath: str):
    filename = os.path.basename(filepath)
    match = MPACK_PATTERN.match(filename)
    if match:
        return match.group(1), int(match.group(2)) if match.group(2) else 18, float(match.group(3))
    jz_match = re.search(r"Jz([0-9.]+)", filename)
    if jz_match:
        return filename.split("_")[0], 18, float(jz_match.group(1))
    return None, None, None

def get_model_instance(ansatz_name: str = "QuantumSelf"):
    ansatz_upper = ansatz_name.upper()
    if "TRANSFORMER" in ansatz_upper or "SELFATT" in ansatz_upper:
        return QuantumSelfAttention(num_layers=4, num_heads=4, param_dtype=jnp.complex128)
    elif "FACTORED" in ansatz_upper:
        return QuantumSelfAttention(num_layers=4, num_heads=4, param_dtype=jnp.complex128)
    return RBM(alpha=2, param_dtype=jnp.complex128)

# =============================================================================
# 4. MOTOR DE EVALUACIÓN GLOBAL (MCMC + EXACT VECTOR)
# =============================================================================
def evaluate_all(
    checkpoint_dir: str = "data/checkpoints",
    output_csv: str = "observables_all_evaluated.csv",
    n_samples: int = 16384
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    mpack_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.mpack")))
    if not mpack_files:
        raise FileNotFoundError(f"No se encontraron checkpoints .mpack en {checkpoint_dir}")

    records_nqs = []
    exact_cache = {}
    
    print(f"\n[INFO] Evaluando {len(mpack_files)} checkpoints NQS con N_samples={n_samples}...")

    for fpath in mpack_files:
        ansatz_name, num_sites, jz = parse_mpack_filename(fpath)
        if jz is None: continue
        
        print(f" -> Procesando: {os.path.basename(fpath)} | Ansatz: {ansatz_name} | Jz: {jz:.2f}")

        # 1. Grafo y Espacio de Hilbert
        graph, hilbert = build_kitaev_lattice(extent=[3,3], pbc=True)
        N = graph.n_nodes
        hilbert = nk.hilbert.Spin(s=0.5, N=N)
        jx = jy = (1 - jz) / 2
        hamiltonian = KitaevTransverse_H(
            graph.edge_colors, graph.edges(), 
            Jx=jx, Jy=jy, Jz=jz, h=0.0, hi=hilbert
        )
        
        # 2. Cargar solución Exacta (ED) para este Jz en caché
        jz_key = round(jz, 3)
        run_exact_diagonalization(extent=[3,3], jz_steps=11, k_eigenvals=1, save_path='data/raw/energies_eigenvecs.npz')
        jz_eigval_eigvec = load_exact_results('data/raw/energies_eigenvecs.npz')
        if jz_key not in exact_cache:
            print(f"    [ED] Calculando Diagonalización Exacta de referencia para Jz={jz:.2f}...")
            jz_eigval_eigvec_temp = jz_eigval_eigvec.get(jz_key)
            e0_ed, psi0_ed = jz_eigval_eigvec_temp['E0'], jz_eigval_eigvec_temp['psi0']
            exact_cache[jz_key] = {"Energy": e0_ed.real / num_sites, "psi0": psi0_ed}
            
            # Observables exactos en ED
            sparse_ops = build_sparse_observables(hilbert, graph)
            for obs_name, op in sparse_ops.items():
                mat_op = op.to_sparse()
                exact_cache[jz_key][obs_name] = np.vdot(psi0_ed, mat_op @ psi0_ed).real
            # Coherencia exacta
            c_l1_ed, _ = calculate_dense_metrics(psi0_ed)
            exact_cache[jz_key]["Cl1"] = c_l1_ed

        # 3. Inicializar NQS y cargar pesos
        model = get_model_instance(ansatz_name)
        sampler = nk.sampler.MetropolisExchange(hilbert, graph=graph, d_max=2, n_chains=16)
        vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)

        with open(fpath, "rb") as file_handle:
            state_dict = flax.serialization.from_bytes(vstate.variables, file_handle.read())
            vstate.variables = state_dict

        row = {
            "file": os.path.basename(fpath), "ansatz": ansatz_name,
            "num_sites": num_sites, "Jz": jz,
        }

        # 4. Evaluación MCMC (Energía y Observables locales)
        energy_stats = vstate.expect(hamiltonian)
        row["Energy"] = energy_stats.mean.real / num_sites
        row["Energy_err"] = energy_stats.error_of_mean / num_sites

        sparse_ops = build_sparse_observables(hilbert, graph)
        for obs_name, op in sparse_ops.items():
            stats = vstate.expect(op)
            row[obs_name] = stats.mean.real
            row[f"{obs_name}_err"] = stats.error_of_mean
            
        # Componente total de magnetización
        row["M_total"] = np.sqrt(row["Mx"]**2 + row["My"]**2 + row["Mz"]**2)

        # 5. Evaluación Densa Exacta (Vector de estado subyacente -> C_l1 y Fidelidad)
        try:
            psi_nqs_dense = vstate.to_array()
            c_l1, fidelity = calculate_dense_metrics(psi_nqs_dense, exact_cache[jz_key]["psi0"])
            row["Cl1"] = c_l1
            row["Fidelity"] = fidelity
            del psi_nqs_dense
            gc.collect()
        except Exception as e:
            print(f"    [WARN] No se pudo proyectar el vector denso para fidelidad: {e}")
            row["Cl1"], row["Fidelity"] = np.nan, np.nan

        records_nqs.append(row)

    df_nqs = pd.DataFrame(records_nqs).sort_values(by=["ansatz", "Jz"])
    df_nqs.to_csv(output_csv, index=False)
    
    # Formatear el DataFrame de ED para graficación
    ed_records = []
    for jz_k, vals in exact_cache.items():
        r = {"Jz": jz_k, **{k: v for k, v in vals.items() if k != "psi0"}}
        r["M_total"] = np.sqrt(r.get("Mx",0)**2 + r.get("My",0)**2 + r.get("Mz",0)**2)
        ed_records.append(r)
    df_ed = pd.DataFrame(ed_records).sort_values(by="Jz")
    
    print(f"\n[SUCCESS] Exportados resultados variacionales a: {output_csv}")
    return df_nqs, df_ed

# =============================================================================
# 5. GENERADOR DE GRÁFICOS MULTI-OBSERVABLE
# =============================================================================
def plot_all_observables(df_nqs: pd.DataFrame, df_ed: pd.DataFrame, output_dir: str = "plots_tfm"):
    os.makedirs(output_dir, exist_ok=True)

    config = [
        ("Energy", "Energía por Sitio $E_0 / N$", "black"),
        ("M_total", "Magnetización Total $\\langle M_{\\text{tot}} \\rangle$", "tab:orange"),
        ("Mx", "Magnetización $\\langle M_x \\rangle$", "tab:blue"),
        ("My", "Magnetización $\\langle M_y \\rangle$", "tab:green"),
        ("Mz", "Magnetización $\\langle M_z \\rangle$", "tab:red"),
        ("Szz", "Correlación de Espín $\\langle S^{zz} \\rangle$", "tab:purple"),
        ("Wp", "Bucle de Wilson $\\langle \\hat{W}_p \\rangle$", "tab:brown"),
        ("Cl1", "Coherencia Cuántica $C_{l_1}$", "tab:cyan"),
        ("Fidelity", "Fidelidad Exacta $|\\langle \\Psi_{\\text{ED}} | \\Psi_{\\text{NQS}} \\rangle|^2$", "crimson")
    ]

    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    ansatz_list = df_nqs["ansatz"].unique()

    for obs_col, ylabel, ed_color in config:
        if obs_col not in df_nqs.columns: continue

        fig, ax = plt.subplots(figsize=(8, 5.5))

        # Línea Teórica Exacta (ED)
        if obs_col in df_ed.columns and obs_col != "Fidelity":
            ax.plot(df_ed["Jz"], df_ed[obs_col], 'o--', color=ed_color, 
                    linewidth=2.0, label="Exacto (ED Lanczos)", zorder=2)
        elif obs_col == "Fidelity":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Límite Teórico Ideal", zorder=2)

        # Puntos Variacionales NQS
        for idx, ansatz_name in enumerate(ansatz_list):
            sub = df_nqs[df_nqs["ansatz"] == ansatz_name].sort_values("Jz")
            y_vals = sub[obs_col]
            y_errs = sub.get(f"{obs_col}_err", np.zeros_like(y_vals))

            ax.errorbar(
                sub["Jz"], y_vals, yerr=y_errs,
                marker=markers[idx % len(markers)], markersize=7, capsize=4,
                linestyle="None", label=f"NQS ({ansatz_name})", zorder=3
            )

        ax.set_xlabel("Anisotropía de Acoplamiento $J_z / J$", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)

        fig.tight_layout()
        save_png = os.path.join(output_dir, f"obs_{obs_col}_vs_Jz.png")
        plt.savefig(save_png)
        plt.close(fig)
        print(f"[EXPORT] Gráfico exportado: {save_png}")

# =============================================================================
# 6. ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    print("="*75)
    print(" PIPELINE DE BENCHMARKING TFM: NQS vs DIAGONALIZACIÓN EXACTA")
    print("="*75)
    
    df_nqs, df_ed = evaluate_all(checkpoint_dir="data/checkpoints", n_samples=16384)
    plot_all_observables(df_nqs, df_ed, output_dir="plots_tfm")
    print("\n[COMPLETE] Análisis comparativo multinivel concluido.")