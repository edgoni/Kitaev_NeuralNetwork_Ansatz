import netket as nk
from typing import Tuple, List, Optional

def build_kitaev_lattice(extent: List[int] = [3, 3], pbc: bool = True) -> Tuple[nk.graph.Graph, nk.hilbert.AbstractHilbert]:
    """
    Construye el grafo honeycomb y el espacio de Hilbert para el modelo de Kitaev.
    
    Args:
        extent: Dimensiones de la red [L1, L2].
        pbc: Condiciones de contorno periódicas (True/False).
        
    Returns:
        Tupla conteniendo el grafo de NetKet y el espacio de Hilbert.
    """
    graph = nk.graph.KitaevHoneycomb(extent=extent, pbc=pbc)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
    
    # Inyectamos el hilbert en el grafo por conveniencia (como en tu código original)
    graph.hi = hilbert 
    
    if extent[0] == 1 or extent[1] == 1:
        import warnings
        warnings.warn("El modelo de Kitaev con una dimensión igual a 1 no está bien definido topológicamente (no hay plaquetas cerradas).")
        
    return graph, hilbert

# Nota: Aquí también deberías migrar tu función `KitaevTransverse_H` 
# que actualmente vive en `utils.py`, para que toda la física del Hamiltoniano 
# esté en un solo lugar.


def KitaevTransverse_H(colores: List[int], 
                       enlaces: List[Tuple[int, int]], 
                       Jx: float, 
                       Jy: float, 
                       Jz: float, 
                       h: float, 
                       hi: nk.hilbert.AbstractHilbert) -> nk.operator.LocalOperator:
    """
    Construye el operador Hamiltoniano de Kitaev con un campo magnético transversal opcional.
    
    Args:
        colores: Lista con la dirección/color de cada enlace (0='x', 1='y', 2='z').
        enlaces: Lista de tuplas indicando los nodos conectados por cada enlace.
        Jx, Jy, Jz: Constantes de acoplamiento para cada dirección.
        h: Intensidad del campo magnético transversal externo.
        hi: Espacio de Hilbert del sistema definido en NetKet.
        
    Returns:
        Operador local (LocalOperator) que representa el Hamiltoniano total.
    """
    # Inicializamos el operador local con tipo complejo
    H = nk.operator.LocalOperator(hi, dtype=complex)
    
    # Pre-cargamos los operadores de Pauli para mayor legibilidad
    sx = nk.operator.spin.sigmax
    sy = nk.operator.spin.sigmay
    sz = nk.operator.spin.sigmaz

    for i, color in enumerate(colores):
        bond = enlaces[i]
        u, v = bond[0], bond[1]
        
        # Término de interacción de Kitaev según el color del enlace
        if color == 0:    # Enlace X
            H -= Jx * (sx(hi, u) @ sx(hi, v))
        elif color == 1:  # Enlace Y
            H -= Jy * (sy(hi, u) @ sy(hi, v))
        elif color == 2:  # Enlace Z
            H -= Jz * (sz(hi, u) @ sz(hi, v))
        else:
            raise ValueError(f"Error: color de enlace {color} no implementado.")

        # Término del campo magnético transversal (opcional, aplicado en [1,1,1])
        if h != 0.0:
            H -= h * (sx(hi, u) + sy(hi, u) + sz(hi, u))

    return H