import argparse
import numpy as np
import netket as nk
import optax  # Añadimos optax para los schedules
from pathlib import Path

# --- Importaciones de tu librería unificada (src) ---
from src.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from src.physics.symmetries import get_kitaev_symmetries
from src.physics.observables import get_kitaev_plaquettes, build_wilson_loops
from src.models.rbm import ProjectedRBM
from src.models.factoredSelfAtt import FactoredAttention, QuantumSelfAttention
from src.training.drivers import setup_vmc_driver
from src.training.callbacks import BestEnergyCheckpoint, build_observables_logger

def main(args):
    print(f"--- Iniciando VMC Pipeline 2-ETAPAS: Modelo={args.model}, Extent={args.L1}x{args.L2} ---")
    
    # 1. Definir la Geometría y el Espacio de Hilbert
    graph, hilbert = build_kitaev_lattice(extent=[args.L1, args.L2], pbc=True)
    N = graph.n_nodes
    
    # 2. Extraer Observables (Plaquetas y Bucles de Wilson)
    plaquetas, ops_colores = get_kitaev_plaquettes(graph)
    Wp_list, Wp_total = build_wilson_loops(hilbert, plaquetas, ops_colores)
    
    # 3. Extraer Simetrías para la Proyección Espacial
    symmetries_info = get_kitaev_symmetries(graph, hilbert)
    space_group = symmetries_info["space_group"]
    
    perms = np.array(graph.automorphisms())
    chars = np.array(space_group.character_table()[args.irrep])

    symm_tuple = tuple(map(tuple, perms.tolist()))
    char_tuple = tuple(chars.tolist())
    
    # 4. MCMC Sampler 
    rule1 = nk.sampler.rules.LocalRule()
    rule2 = nk.sampler.rules.ExchangeRule(graph=graph)
    sampler = nk.sampler.MetropolisSampler(
        hilbert, 
        rule=nk.sampler.rules.MultipleRules([rule1, rule2], [0.9, 0.1]),
        n_chains=args.n_chains
    )

    jz_values = np.linspace(args.jz_start, args.jz_end, args.jz_steps)
    Path("data/checkpoints").mkdir(parents=True, exist_ok=True)

    transfer_params = None

    # 5. Bucle Principal de Entrenamiento y Transfer Learning
    for jz in jz_values:
        print(f"\n" + "="*40)
        print(f">>> Entrenando para Jz = {jz:.2f} <<<")
        print("="*40)
        
        jx = jy = (1 - jz) / 2
        H = KitaevTransverse_H(
            graph.edge_colors, graph.edges(), 
            Jx=jx, Jy=jy, Jz=jz, h=0.0, hi=hilbert
        )

        # ===================================================================
        # ETAPA 1: Entrenamiento SIN Proyección (Warm-up general)
        # ===================================================================
        print(f"\n--- ETAPA 1: Sin Proyección (Warm-up {args.n_iter_1} iteraciones) ---")
        if args.model == "RBM":
            model_stage1 = ProjectedRBM(alpha=args.alpha, symmetries=None, characters=None)
        elif args.model == "Transformer":
            model_stage1 = QuantumSelfAttention(layers=args.layers, heads=args.heads, symmetries=None, characters=None)
            
        vstate_s1 = nk.vqs.MCState(sampler, model_stage1, n_samples=args.n_samples)
        vstate_s1.chunk_size = 128
        
        if transfer_params is not None:
            vstate_s1.parameters = transfer_params

        # --- SCHEDULE DE LEARNING RATE PARA ETAPA 1 ---
        # Decae linealmente desde el LR inicial hasta un 10% del mismo
        lr_schedule_s1 = optax.linear_schedule(
            init_value=args.learning_rate,
            end_value=args.learning_rate * 0.1,
            transition_steps=args.n_iter_1
        )

        driver_s1 = setup_vmc_driver(vstate_s1, H, learning_rate=lr_schedule_s1, use_sr=args.use_sr)
        
        metrics_s1 = {'step': [], 'energy': [], 'energy_error': [], 'variance': [], 'wp_mean': []}
        ckpt_path_s1 = Path(f"data/checkpoints/{args.exp_name}_Jz{jz:.2f}_Stage1.mpack")
        checkpoint_s1 = BestEnergyCheckpoint(H, save_path=ckpt_path_s1)
        logger_s1 = build_observables_logger(metrics_s1, H, wp_operators=Wp_list)
        tb_logger_s1 = nk.logging.TensorBoardLog(f"data/tb_logs/{args.exp_name}_Jz{jz:.2f}_Stage1")

        driver_s1.run(n_iter=args.n_iter_1, out=tb_logger_s1, callback=[checkpoint_s1, logger_s1], show_progress=True)
        
        # ===================================================================
        # ETAPA 2: Entrenamiento CON Proyección (Colapso al sector topológico)
        # ===================================================================
        #tenemos que acceder a los GS,  
        data_energy = np.load('data/raw/energies_eigenvecs.npz', allow_pickle=True)
        exact_results_dict = data_energy['data_dict'].item()
        
        # Generar la clave exactamente con el mismo formato numérico (float) 
        # y precisión (4 decimales) con el que fue guardado en run_exact_diagonalization
        jz_key = round(float(jz), 4)
        
        # Acceder a las contribuciones de las representaciones irreducibles
        irrep_contributions_dict = exact_results_dict[jz_key]['irrep_contributions']
        
        # Identificar la irrep dominante
        best_irrep_str = max(irrep_contributions_dict, key=irrep_contributions_dict.get)
        args.irrep = int(best_irrep_str) 
        max_contribution = irrep_contributions_dict[best_irrep_str]
        
        print(f"[ED Info] Para Jz={jz_key}, la irrep dominante es {args.irrep} con peso {max_contribution:.4f}")

        print(f"Física del Estado Fundamental: Para J_z={jz:.1f}, el sector dominante es la Irrep {args.irrep} "
              f"(Contribución: {max_contribution:.4f}).")

        if args.use_symmetry:
            print(f"\n--- ETAPA 2: Con Proyección Irrep {args.irrep} ({args.n_iter_2} iteraciones) ---")
            if args.model == "RBM":
                model_stage2 = ProjectedRBM(alpha=args.alpha, symmetries=symm_tuple, characters=char_tuple)
            elif args.model == "Transformer":
                model_stage2 = QuantumSelfAttention(layers=args.layers, heads=args.heads, symmetries=symm_tuple, characters=char_tuple)
                
            vstate_s2 = nk.vqs.MCState(sampler, model_stage2, n_samples=args.n_samples)
            vstate_s2.chunk_size = 128
            
            # Transferencia de pesos
            best_s1_params = checkpoint_s1.best_state_params if checkpoint_s1.best_state_params is not None else vstate_s1.parameters
            vstate_s2.parameters = best_s1_params

            # --- SCHEDULE DE LEARNING RATE PARA ETAPA 2 ---
            # Arranca donde terminó la Etapa 1 y decae hasta un valor residual (ej: 1% del inicial)
            # Esto evita que el QNG / SR destruya los pesos de la primera etapa y solo afine.
            lr_schedule_s2 = optax.linear_schedule(
                init_value=args.learning_rate * 0.1,
                end_value=args.learning_rate * 0.01, 
                transition_steps=args.n_iter_2
            )

            driver_s2 = setup_vmc_driver(vstate_s2, H, learning_rate=lr_schedule_s2, use_sr=args.use_sr)
            
            metrics_s2 = {'step': [], 'energy': [], 'energy_error': [], 'variance': [], 'wp_mean': []}
            ckpt_path_s2 = Path(f"data/checkpoints/{args.exp_name}_Jz{jz:.2f}_Stage2.mpack")
            checkpoint_s2 = BestEnergyCheckpoint(H, save_path=ckpt_path_s2)
            logger_s2 = build_observables_logger(metrics_s2, H, wp_operators=Wp_list)
            tb_logger_s2 = nk.logging.TensorBoardLog(f"data/tb_logs/{args.exp_name}_Jz{jz:.2f}_Stage2")

            driver_s2.run(n_iter=args.n_iter_2, out=tb_logger_s2, callback=[checkpoint_s2, logger_s2], show_progress=True)
            
            transfer_params = checkpoint_s2.best_state_params if checkpoint_s2.best_state_params is not None else vstate_s2.parameters
        else:
            print("\n--- (Saltando Etapa 2 porque --use_symmetry no está activado) ---")
            transfer_params = checkpoint_s1.best_state_params if checkpoint_s1.best_state_params is not None else vstate_s1.parameters

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
    
    # Hiperparámetros
    parser.add_argument("--alpha", type=float, default=1.0, help="Densidad RBM")
    parser.add_argument("--layers", type=int, default=2, help="Capas del Transformer")
    parser.add_argument("--heads", type=int, default=4, help="Cabezales del Transformer")
    
    # MCMC y Entrenamiento
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--n_iter_1", type=int, default=300)
    parser.add_argument("--n_iter_2", type=int, default=300)
    parser.add_argument("--n_chains", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--use_sr", action="store_true", help="Usar Stochastic Reconfiguration (QNG)")
    
    args = parser.parse_args()
    main(args)