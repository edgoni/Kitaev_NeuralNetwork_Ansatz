import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional

class FactoredAttention(nn.Module):
    """
    Implementación de la 'Factored Attention' basada en arXiv:2405.18874v2.
    Elimina Queries y Keys, utilizando únicamente una matriz de pesos 
    posicionales relativos entrenables y una proyección de Values.
    """
    num_heads: int
    head_dim: int
    param_dtype: Any = jnp.complex128

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: (..., seq_len, d_model)
        seq_len = x.shape[-2]
        
        # 1. Proyección de Valores (Values) - Equivalente a V * x_j
        v = nn.Dense(
            features=self.num_heads * self.head_dim, 
            use_bias=False, 
            param_dtype=self.param_dtype,
            name='value_proj'
        )(x)
        # Reshape para separar las cabezas: (..., seq_len, num_heads, head_dim)
        v = v.reshape(*x.shape[:-2], seq_len, self.num_heads, self.head_dim)
        
        # 2. Pesos Posicionales (Input-independent Attention Maps)
        # Creamos una matriz de parámetros P de tamaño (num_heads, seq_len, seq_len)
        # Esto equivale a p_{i-j} en la Ecuación 6 del paper.
        rel_pos_bias = self.param(
            'rel_pos_bias',
            nn.initializers.normal(stddev=0.02), # Inicialización pequeña
            (self.num_heads, seq_len, seq_len),
            self.param_dtype
        )
        
        # 3. Aplicar la Atención Factorizada
        # A_i = sum_j p_{i,j} * V * x_j
        # Utilizamos einsum para contraer la matriz posicional con los valores
        # 'hij' = heads, seq_len(i), seq_len(j)
        # '...jhd' = batch, seq_len(j), heads, head_dim
        # Resultado '...ihd' = batch, seq_len(i), heads, head_dim
        out = jnp.einsum('hij,...jhd->...ihd', rel_pos_bias, v)
        
        # 4. Concatenar cabezas y Proyección Final (W)
        out = out.reshape(*x.shape[:-2], seq_len, self.num_heads * self.head_dim)
        out = nn.Dense(
            features=x.shape[-1], 
            use_bias=False, 
            param_dtype=self.param_dtype,
            name='out_proj'
        )(out)
        
        return out


class QuantumSelfAttention(nn.Module):
    layers: int = 2
    heads: int = 4
    dk: int = 4 # head_dim
    param_dtype: Any = jnp.complex128
    
    symmetries: Optional[tuple] = None
    characters: Optional[tuple] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Añadimos una dimensión al final si los espines son escalares (..., N) -> (..., N, 1)
        if x.ndim == 1 or (x.ndim == 2 and x.shape[0] != x.shape[-1]): 
             h = jnp.expand_dims(x, axis=-1)
        else:
             h = x

        # Embedding lineal (d_model = heads * dk)
        d_model = self.heads * self.dk
        h = nn.Dense(features=d_model, param_dtype=self.param_dtype, name='embedding')(h)

        N = h.shape[-2] # Número de nodos (espines)

        # Bloques del Transformer
        for i in range(self.layers):
            # Capa de Atención Factorizada (Reemplaza a nn.SelfAttention)
            attn_out = FactoredAttention(
                num_heads=self.heads, 
                head_dim=self.dk, 
                param_dtype=self.param_dtype,
                name=f'factored_attn_{i}'
            )(h)
            
            h = h + attn_out
            h = nn.LayerNorm(name=f'ln1_{i}', param_dtype=self.param_dtype)(h)

            # Feed Forward Network (MLP)
            mlp = nn.Dense(features=d_model * 4, name=f'mlp_up_{i}', param_dtype=self.param_dtype)(h)
            mlp = nn.gelu(mlp)
            mlp = nn.Dense(features=d_model, name=f'mlp_down_{i}', param_dtype=self.param_dtype)(mlp)

            h = h + mlp
            h = nn.LayerNorm(name=f'ln2_{i}', param_dtype=self.param_dtype)(h)

        # Pooling y Proyección final a log-amplitudes
        h_flat = h.reshape(*h.shape[:-2], N * d_model)
        log_amps = nn.Dense(features=1, name='output_proj', param_dtype=self.param_dtype)(h_flat)
        log_amps = jnp.squeeze(log_amps, axis=-1)

        # Colapso Final al Sector de Simetría (Igual que en la RBM)
        if self.symmetries is not None and self.characters is not None:
            symm_jnp = jnp.array(self.symmetries)
            chars_conj = jnp.conj(jnp.array(self.characters))
            
            # (Aquí iría tu lógica exacta de proyección usando symm_jnp)
            # Para evitar errores devuelvo log_amps directo como placeholder
            pass 
            
        return log_amps