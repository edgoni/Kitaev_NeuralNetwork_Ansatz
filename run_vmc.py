import argparse
import numpy as np
import netket as nk
from pathlib import Path

# --- Importaciones de tu librería unificada (src) ---
from src.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from src.physics.symmetries import get_kitaev_symmetries
from src.physics.observables import get_kitaev_plaquettes, build_wilson_loops
from src.models.rbm import ProjectedRBM
#from models.quantumself import QuantumSelfAttention
from src.models.factoredSelfAtt import FactoredAttention, QuantumSelfAttention
from src.training.drivers import setup_vmc_driver
from src.training.callbacks import BestEnergyCheckpoint, build_observables_logger

def main(args):
    print(f"--- Iniciando VMC Pipeline: Modelo={args.model}, Extent={args.L1}x{args.L2} ---")
    
    # 1. Definir la Geometría y el Espacio de Hilbert
    graph, hilbert = build_kitaev_lattice(extent=[args.L1, args.L2], pbc=True)
    N = graph.n_nodes
    
    # 2. Extraer Observables (Plaquetas y Bucles de Wilson)
    plaquetas, ops_colores = get_kitaev_plaquettes(graph)
    Wp_list, Wp_total = build_wilson_loops(hilbert, plaquetas, ops_colores)
    
    # 3. Extraer Simetrías para la Proyección Espacial
    symmetries_info = get_kitaev_symmetries(graph, hilbert)
    space_group = symmetries_info["space_group"]
    
    # Array de permutaciones del grupo espacial y caracteres del Irrep seleccionado
    perms = np.array(graph.automorphisms())
    chars = np.array(space_group.character_table()[args.irrep])

    symm_tuple = tuple(map(tuple, perms.tolist()))
    char_tuple = tuple(chars.tolist())
    
    # 4. Instanciar el Ansatz Variacional (JAX/Flax)
    if args.model == "RBM":
        vstate_model = ProjectedRBM(
            alpha=args.alpha, 
            symmetries=symm_tuple if args.use_symmetry else None, 
            characters=char_tuple if args.use_symmetry else None
        )
    elif args.model == "Transformer":
        vstate_model = QuantumSelfAttention(
            layers=args.layers, 
            heads=args.heads,
            symmetries=symm_tuple if args.use_symmetry else None, 
            characters=char_tuple if args.use_symmetry else None
        )
    else:
        raise ValueError(f"Modelo {args.model} no soportado.")

    # 5. MCMC Sampler (Combinación de Local Flips y Exchange para mayor ergodicidad)
    rule1 = nk.sampler.rules.LocalRule()
    rule2 = nk.sampler.rules.ExchangeRule(graph=graph)
    sampler = nk.sampler.MetropolisSampler(
        hilbert, 
        rule=nk.sampler.rules.MultipleRules([rule1, rule2], [0.9, 0.1]),
        n_chains=args.n_chains
    )

    # 6. Estado Variacional (Instanciado FUERA del bucle para Transfer Learning)
    # --- EN run_vmc.py ---
    vstate = nk.vqs.MCState(sampler, vstate_model, n_samples=args.n_samples)

    # Añadir esto para evitar el Out Of Memory (OOM) en el cálculo del Jacobiano
    vstate.chunk_size = 128  # Si sigue fallando, bájalo a 64

    # Preparar el barrido de parámetros Jz
    jz_values = np.linspace(args.jz_start, args.jz_end, args.jz_steps)
    Path("data/checkpoints").mkdir(parents=True, exist_ok=True)

    # 7. Bucle Principal de Entrenamiento y Transfer Learning
    for jz in jz_values:
        print(f"\n>>> Entrenando para Jz = {jz:.2f} <<<")
        
        # Construir el Hamiltoniano específico para este Jz
        jx = jy = (1 - jz) / 2
        H = KitaevTransverse_H(
            graph.edge_colors, graph.edges(), 
            Jx=jx, Jy=jy, Jz=jz, h=0.0, hi=hilbert
        )

        # Configurar el Driver (Optimizador y SR)
        driver = setup_vmc_driver(
            vstate, H, 
            learning_rate=args.learning_rate, 
            use_sr=args.use_sr
        )

        # Preparar métricas y Callbacks
        metrics_history = {'step': [], 'energy': [], 'energy_error': [], 'variance': [], 'wp_mean': []}
        
        ckpt_path = Path(f"data/checkpoints/{args.exp_name}_Jz{jz:.2f}.mpack")
        checkpoint = BestEnergyCheckpoint(H, save_path=ckpt_path)
        logger_cb = build_observables_logger(metrics_history, H, wp_operators=Wp_list)

        # Configurar Logger nativo de NetKet
        tensorboard_logger = nk.logging.TensorBoardLog(f"data/tb_logs/{args.exp_name}_Jz{jz:.2f}")

        # Ejecutar Entrenamiento
        driver.run(
            n_iter=args.n_iter, 
            out=tensorboard_logger,
            callback=[checkpoint, logger_cb],
            show_progress=True
        )

        # --- TRANSFER LEARNING ADIABÁTICO ---
        # Cargamos los mejores parámetros de este Jz para que sean la inicialización
        # del próximo Jz. Esto es crucial para seguir la evolución del Ground State.
        if checkpoint.best_state_params is not None:
            vstate.parameters = checkpoint.best_state_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar VMC para Kitaev")
    parser.add_argument("--exp_name", type=str, required=True, help="Nombre del experimento")
    parser.add_argument("--model", type=str, choices=["RBM", "Transformer"], default="RBM")
    
    # Geometría y Física
    parser.add_argument("--L1", type=int, default=3)
    parser.add_argument("--L2", type=int, default=3)
    parser.add_argument("--jz_start", type=float, default=0.0)
    parser.add_argument("--jz_end", type=float, default=1.0)
    parser.add_argument("--jz_steps", type=int, default=11)
    parser.add_argument("--irrep", type=int, default=0, help="Índice Irrep (0 = Ground State free of vortices)")
    parser.add_argument("--use_symmetry", action="store_true", help="Activar la proyección de simetría espacial")
    
    # Hiperparámetros de los Modelos
    parser.add_argument("--alpha", type=float, default=1.0, help="Densidad RBM")
    parser.add_argument("--layers", type=int, default=2, help="Capas del Transformer")
    parser.add_argument("--heads", type=int, default=4, help="Cabezales del Transformer")
    
    # MCMC y Entrenamiento
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument("--n_chains", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--use_sr", action="store_true", help="Usar Stochastic Reconfiguration (QNG)")
    
    args = parser.parse_args()
    main(args)