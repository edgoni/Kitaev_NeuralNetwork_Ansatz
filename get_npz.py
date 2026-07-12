import argparse
import numpy as np
import netket as nk
import optax  # Añadimos optax para los schedules
from pathlib import Path
import os

# --- Importaciones de tu librería unificada (src) ---
from src.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from src.physics.symmetries import get_kitaev_symmetries
from src.physics.observables import get_kitaev_plaquettes, build_wilson_loops
from src.physics.exact_diag import run_exact_diagonalization, identify_irreps
from src.models.rbm import ProjectedRBM
from src.models.factoredSelfAtt import FactoredAttention, QuantumSelfAttention
from src.training.drivers import setup_vmc_driver
from src.training.callbacks import BestEnergyCheckpoint, build_observables_logger


graph, hilbert = build_kitaev_lattice(extent=[3, 3], pbc=True)
N = graph.n_nodes

jz_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for jz in jz_values:
        print(f"\n" + "="*40)
        print(f">>> Entrenando para Jz = {jz:.2f} <<<")
        print("="*40)
        
        jx = jy = (1 - jz) / 2
        H = KitaevTransverse_H(
            graph.edge_colors, graph.edges(), 
            Jx=jx, Jy=jy, Jz=jz, h=0.0, hi=hilbert
        )

        run_exact_diagonalization(extent=[3,3], jz_steps=11, k_eigenvals=5, save_path='data/raw/energies_eigenvecs.npz', save_debug_json=True)
                

        