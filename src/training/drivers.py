import netket as nk
import optax
from typing import Any, Optional

def setup_vmc_driver(vstate: nk.vqs.MCState, 
                     hamiltonian: Any, 
                     learning_rate: Any = 0.01, 
                     use_sr: bool = True,
                     diag_shift: float = 0.01) -> nk.driver.VMC:
    """
    Configura y devuelve el driver Variational Monte Carlo (VMC).
    
    Args:
        vstate: Estado variacional de NetKet.
        hamiltonian: Operador del Hamiltoniano local.
        learning_rate: Tasa de aprendizaje (puede ser un flotante o un optax schedule).
        use_sr: Si es True, utiliza Stochastic Reconfiguration (Quantum Natural Gradient).
        diag_shift: Desplazamiento en la diagonal de la matriz S de SR para estabilidad numérica.
        
    Returns:
        Un objeto nk.driver.VMC listo para ejecutar .run().
    """
    
    # 1. Optimizador (Soportamos Schedulers de Optax directamente)
    # NetKet 3 soporta nativamente optimizadores Optax (AdaGrad, Adam, SGD)
    if isinstance(learning_rate, float):
        optimizer = optax.adagrad(learning_rate)
    else:
        # Asume que es un callable de schedule (como optax.warmup_exponential_decay_schedule)
        optimizer = optax.adagrad(learning_rate)

    # 2. Reconfiguración Estocástica (Precondicionador)
    preconditioner = None
    if use_sr:
        # qgt_options define el solver. nk.optimizer.qgt.QGTJacobianDense 
        # es muy estable para sistemas intermedios (N=18).
        preconditioner = nk.optimizer.SR(
            diag_shift=diag_shift, 
            qgt=nk.optimizer.qgt.QGTJacobianDense
        )

    # 3. Ensamblar el Driver
    driver = nk.driver.VMC(
        hamiltonian, 
        optimizer, 
        variational_state=vstate, 
        preconditioner=preconditioner
    )
    
    return driver