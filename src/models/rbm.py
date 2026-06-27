import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from typing import Any, Optional

class ProjectedRBM(nn.Module):
    """
    Ansatz basado en Deep RBM para sistemas de espines entrelazados.
    Permite la proyección explícita a sectores de simetría espacial (Irreps).
    
    Attributes:
        num_layers: Número de capas ocultas.
        alpha: Densidad de la red (n_hidden = alpha * N).
        param_dtype: Tipo de dato de los pesos (jnp.complex128 recomendado para NQS).
        symmetries: Array de permutaciones del grupo espacial de tamaño (n_g, N).
        characters: Array de caracteres del irrep deseado de tamaño (n_g,).
    """
    num_layers: int = 2
    alpha: float = 1.0
    param_dtype: Any = jnp.complex128
    
    symmetries: Optional[jnp.ndarray] = None
    characters: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evalúa el logaritmo de la amplitud de la función de onda ψ(x).
        
        Args:
            x: Configuraciones de espín. Shape (..., N).
            
        Returns:
            Logaritmo de las amplitudes proyectadas. Shape (...).
        """
        N = x.shape[-1]
        
        # 1. Transformación Espacial (Proyección)
        if self.symmetries is not None:
            # Si hay simetría, evaluamos todas las permutaciones simultáneamente
            # Shape resultante: (..., n_g, N)
            x_eval = x[..., self.symmetries] 
        else:
            # Shape resultante: (..., N)
            x_eval = x
            
        # 2. Forward Pass (Broadcasting nativo de Flax)
        h = x_eval
        kernel_init = nn.initializers.normal(stddev=0.01)
        bias_init = nn.initializers.normal(stddev=0.1)

        for i in range(self.num_layers):
            n_hidden = int(self.alpha * N)
            h = nn.Dense(
                features=n_hidden,
                use_bias=True,
                param_dtype=self.param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name=f"layer_{i}"
            )(h)
            
            # LayerNorm estabiliza el gradiente en redes NQS profundas
            h = nn.LayerNorm(
                param_dtype=self.param_dtype,
                use_scale=False,
                use_bias=False,
                name=f"ln_{i}"
            )(h)
            h = nk.nn.log_cosh(h)

        # Colapso a log-amplitudes. Shape resultante: (...) o (..., n_g)
        res = jnp.sum(h, axis=-1)

        # Bias visible (Campo local acoplado directamente a los espines)
        v_bias = self.param(
            "visible_bias", 
            bias_init, 
            (N,), 
            self.param_dtype
        )
        
        # El producto punto se adapta automáticamente al shape de x_eval
        out_bias = jnp.dot(x_eval, v_bias)
        
        log_amps = res + out_bias

        # 3. Colapso Final al Sector de Simetría
        if self.symmetries is not None and self.characters is not None:
            chars_conj = jnp.conj(jnp.array(self.characters))
            
            # Log-Sum-Exp Trick para evitar desbordamiento en e^(log_amps)
            log_max = jnp.max(jnp.real(log_amps), axis=-1, keepdims=True)
            amps_rel = jnp.exp(log_amps - log_max)
            
            # Suma ponderada: Σ_g χ*(g) * ψ_rel(P_g x)
            weighted = jnp.sum(chars_conj * amps_rel, axis=-1)
            
            # Restaurar la escala global y calcular el logaritmo final
            log_proj = jnp.log(weighted) + log_max[..., 0]
            return log_proj
            
        return log_amps