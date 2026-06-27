import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional

class QuantumSelfAttention(nn.Module):
    """
    Ansatz Variacional basado en Transformers (Self-Attention) para estados cuánticos.
    
    Físicamente, a diferencia de la RBM que restringe la conectividad a una topología
    bipartita, el mecanismo de Auto-Atención (Self-Attention) captura correlaciones 
    cuánticas de largo alcance de forma al-to-all sin restricciones espaciales. 
    Esto es críticamente ventajoso para describir el régimen sin gap (gapless spin liquid)
    del modelo de Kitaev, donde las fluctuaciones no locales dominan la función de onda.
    
    Attributes:
        layers: Número de bloques Transformer.
        heads: Número de cabezales de atención.
        dk: Dimensión del espacio latente por cabezal.
        param_dtype: Tipo complejo para funciones de onda.
        symmetries: Permutaciones del grupo espacial.
        characters: Caracteres del irrep proyectado.
    """
    layers: int = 2
    heads: int = 4
    dk: int = 4
    param_dtype: Any = jnp.complex128
    
    symmetries: Optional[jnp.ndarray] = None
    characters: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        N = x.shape[-1]
        
        # 1. Transformación Espacial (Proyección del Grupo)
        if self.symmetries is not None:
            # Shape resultante: (..., n_g, N)
            x_eval = x[..., self.symmetries]  
        else:
            # Shape resultante: (..., N)
            x_eval = x 
            
        # 2. Preparación de la secuencia (Treating spins as "tokens")
        # Añadimos dimensión de features: Shape -> (..., N, 1)
        h = jnp.expand_dims(x_eval, axis=-1) 
        
        d_model = self.heads * self.dk
        h = nn.Dense(
            features=d_model, 
            name='embedding', 
            param_dtype=self.param_dtype
        )(h) # Shape -> (..., N, d_model)

        # 3. Bloques del Transformer
        for i in range(self.layers):
            # SelfAttention nativo de Flax opera automáticamente sobre el eje -2 (secuencia N)
            # sin importar cuántas dimensiones de batch (como n_g) haya por delante.
            att_out = nn.SelfAttention(
                num_heads=self.heads, 
                qkv_features=d_model, 
                name=f'att_{i}',
                param_dtype=self.param_dtype
            )(h)
            
            # Conexión residual + LayerNorm
            h = h + att_out
            h = nn.LayerNorm(name=f'ln1_{i}', param_dtype=self.param_dtype)(h)

            # Feed Forward Network (MLP)
            mlp = nn.Dense(features=d_model * 4, name=f'mlp_up_{i}', param_dtype=self.param_dtype)(h)
            mlp = nn.gelu(mlp)
            mlp = nn.Dense(features=d_model, name=f'mlp_down_{i}', param_dtype=self.param_dtype)(mlp)

            h = h + mlp
            h = nn.LayerNorm(name=f'ln2_{i}', param_dtype=self.param_dtype)(h)

        # 4. Pooling / Flattening robusto
        # Aplanamos conservando todas las dimensiones de batch intactas.
        # Pasa de (..., N, d_model) a (..., N * d_model)
        h_flat = h.reshape(*h.shape[:-2], N * d_model)
        
        # Proyección final a log-amplitud (escalar complejo)
        log_amps = nn.Dense(features=1, name='output_proj', param_dtype=self.param_dtype)(h_flat)
        log_amps = jnp.squeeze(log_amps, axis=-1) # Shape -> (...) o (..., n_g)

        # 5. Colapso Final al Sector de Simetría (Log-Sum-Exp Trick)
        if self.symmetries is not None and self.characters is not None:
            chars_conj = jnp.conj(jnp.array(self.characters))
            
            log_max = jnp.max(jnp.real(log_amps), axis=-1, keepdims=True)
            amps_rel = jnp.exp(log_amps - log_max)
            
            weighted = jnp.sum(chars_conj * amps_rel, axis=-1)
            log_proj = jnp.log(weighted) + log_max[..., 0]
            
            return log_proj
            
        return log_amps