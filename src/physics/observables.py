from collections import defaultdict
from typing import List, Tuple
from netket.operator.spin import sigmaz, sigmax, sigmay
import netket as nk

def get_kitaev_plaquettes(graph: nk.graph.Graph) -> Tuple[List[List[int]], List[List[str]]]:
    """
    Extrae las plaquetas (ciclos hexagonales) y los operadores Wp asociados.
    """
    edges = list(graph.edges())
    colors = list(graph.edge_colors)
    cmap = {0: 'x', 1: 'y', 2: 'z'}
    todos = {'x', 'y', 'z'}

    vecinos_color = defaultdict(lambda: defaultdict(list))
    for (u, v), c in zip(edges, colors):
        col = cmap[c]
        vecinos_color[u][col].append(v)
        vecinos_color[v][col].append(u)

    plaquetas = []
    operadores = []
    visitadas = set()

    for start_node in range(graph.n_nodes):
        for c0 in ['x', 'y', 'z']:
            if not vecinos_color[start_node][c0]:
                continue

            for seq in _hexagon_sequences(c0):
                for v0 in vecinos_color[start_node][seq[0]]:
                    ciclo = _build_cycle(start_node, seq, vecinos_color)
                    if ciclo is None:
                        continue
                    key = tuple(sorted(ciclo))
                    if key in visitadas:
                        continue
                        
                    ops = _calculate_ops(ciclo, seq, todos)
                    visitadas.add(key)
                    plaquetas.append(ciclo)
                    operadores.append(ops)

    return plaquetas, operadores

def build_wilson_loops(hi: nk.hilbert.AbstractHilbert, plaquetas: List[List[int]], ops_colores: List[List[str]]) -> Tuple[List[nk.operator.LocalOperator], nk.operator.LocalOperator]:
    """
    Construye los operadores locales de Wilson loop (Wp) para cada plaqueta.
    """
    mapa_sigmas = {'x': sigmax, 'y': sigmay, 'z': sigmaz}
    Wp_list = []

    for p_idx, nodos in enumerate(plaquetas):
        ops_p = ops_colores[p_idx]
        Wp = mapa_sigmas[ops_p[0]](hi, nodos[0])

        for site, op_char in zip(nodos[1:], ops_p[1:]):
            Wp = Wp @ mapa_sigmas[op_char](hi, site)

        Wp_list.append(Wp)

    Wp_total = sum(Wp_list) / len(Wp_list) if Wp_list else None
    return Wp_list, Wp_total

# --- Funciones privadas (ocultas para mantener la API limpia) ---
def _hexagon_sequences(c0: str) -> List[List[str]]:
    otros = [c for c in ['x', 'y', 'z'] if c != c0]
    return [
        [c0, otros[0], otros[1], c0, otros[0], otros[1]],
        [c0, otros[1], otros[0], c0, otros[1], otros[0]],
    ]

def _build_cycle(start: int, seq_colores: List[str], vecinos_color: dict) -> List[int]:
    ciclo = [start]
    nodo_actual = start
    for i, col in enumerate(seq_colores):
        vecinos = vecinos_color[nodo_actual][col]
        candidatos = [v for v in vecinos if v not in ciclo] if i < 5 else [v for v in vecinos if v == start]
        if not candidatos: return None
        nodo_actual = candidatos[0]
        if i < 5: ciclo.append(nodo_actual)
    return ciclo if nodo_actual == start else None

def _calculate_ops(ciclo: List[int], seq_colores: List[str], todos: set) -> List[str]:
    n = len(ciclo)
    return [(todos - {seq_colores[(i - 1) % n], seq_colores[i]}).pop() for i in range(n)]