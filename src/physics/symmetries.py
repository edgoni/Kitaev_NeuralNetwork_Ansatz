import netket as nk
from typing import Dict, Any

def get_kitaev_symmetries(graph: nk.graph.Graph, hilbert: nk.hilbert.AbstractHilbert) -> Dict[str, Any]:
    """
    Extrae la información geométrica y de simetría (grupo espacial, irreps) de la red.
    """
    space_group = graph.space_group()
    
    return {
        "canonical_rep": nk.symmetry.canonical_representation(hilbert, graph.translation_group()),
        "translations": graph.translation_group(),
        "automorphisms": graph.automorphisms(),
        "point_group": nk.utils.group.PointGroup(graph.point_group(), ndim=2),
        "space_group": space_group,
        "irreps_matrices": space_group.irrep_matrices(),
        "character_table": space_group.character_table()
    }