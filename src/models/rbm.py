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
    
class DeepRBMSymmProj(nn.Module):
    """
    DeepRBM con proyección explícita al sector de simetría,
    equivalente a lo que hace el repositorio TQC con self.symmetry.mm(bx).
    
    La proyección se aplica DESPUÉS de calcular log|ψ(σ)|,
    construyendo la amplitud proyectada como:
        ψ_proj(σ) = Σ_g  χ*(g) * ψ(P_g σ)
    que es exactamente S @ ψ en espacio de configuraciones.
    """
    num_layers:  int = 2
    alpha:       float = 1.0
    param_dtype: Any = jnp.complex128
    # Tabla de permutaciones: shape (n_group, n_sites)
    # Cada fila es una permutación del grupo de simetría
    symmetries:  Any = None   # HashableArray de shape (n_g, N)
    # Caracteres del irrep deseado: shape (n_group,)
    characters:  Any = None   # HashableArray de shape (n_g,)

    @nn.compact
    def __call__(self, x):
        """
        x: configuración de espines, shape (..., N) con valores ±1
        """
        input_spins = x

        kernel_init = nn.initializers.normal(stddev=0.01)
        bias_init   = nn.initializers.normal(stddev=0.1)

        def _rbm_logpsi(spin_config):
            """Aplica las capas RBM a una configuración dada."""
            h = spin_config
            for i in range(self.num_layers):
                n_hidden = int(self.alpha * h.shape[-1])
                h = nn.Dense(
                    features=n_hidden,
                    use_bias=True,
                    param_dtype=self.param_dtype,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name=f"layer_{i}"
                )(h)
                h = nn.LayerNorm(
                    param_dtype=self.param_dtype,
                    use_scale=False,
                    use_bias=False,
                    name=f"ln_{i}"
                )(h)
                h = nk.nn.log_cosh(h)

            res = jnp.sum(h, axis=-1)

            v_bias = self.param(
                "visible_bias",
                bias_init,
                (spin_config.shape[-1],),
                self.param_dtype,
            )
            return res + jnp.dot(spin_config, v_bias)

        if self.symmetries is None or self.characters is None:
            # Sin simetría: comportamiento idéntico a DeepRBM
            return _rbm_logpsi(input_spins)

        # -------------------------------------------------------
        # Proyección al sector de simetría
        # ψ_proj(σ) = log[ Σ_g χ*(g) * exp(log_ψ(P_g σ)) ]
        #
        # Equivalente a S @ ψ del repositorio TQC, pero en log-space
        # para evitar overflow numérico.
        # -------------------------------------------------------
        perms  = jnp.array(self.symmetries)   # (n_g, N)
        chars  = jnp.array(self.characters)   # (n_g,)  complejo

        # Aplica cada permutación g a la configuración σ
        # x_perm[g] = P_g σ  →  shape (n_g, N)
        x_perm = input_spins[..., perms]      # (..., n_g, N)

        # Calcula log_ψ(P_g σ) para cada g en paralelo con vmap
        # Usamos nn.vmap si estamos en modo batch, o jax.vmap manualmente

        # log_amps[g] = log_ψ(P_g σ),  shape (..., n_g)
        log_amps = jax.vmap(
            _rbm_logpsi,
            in_axes=0,          # mapea sobre el eje de permutaciones
            out_axes=0
        )(x_perm.reshape(-1, input_spins.shape[-1]))  # (n_g * batch, N)

        # Reshapear a (..., n_g)
        batch_shape = input_spins.shape[:-1]
        n_g = perms.shape[0]
        log_amps = log_amps.reshape(*batch_shape, n_g)  # (..., n_g)

        # log[ Σ_g χ*(g) * exp(log_ψ(P_g σ)) ]
        chars_conj = jnp.conj(chars)  # (n_g,)

        #sustraer el máximo para estabilidad
        log_max   = jnp.max(jnp.real(log_amps), axis=-1, keepdims=True)
        amps_rel  = jnp.exp(log_amps - log_max)                    # (..., n_g)
        weighted  = jnp.sum(chars_conj * amps_rel, axis=-1)        # (...)
        log_proj  = jnp.log(weighted) + log_max[..., 0]            # (...)

        return log_proj