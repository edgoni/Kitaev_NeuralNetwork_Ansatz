import flax
import copy
import numpy as np
import jax.numpy as jnp
from typing import Optional, Dict, List, Any
import pathlib
import numpy.typing as npt

class BestEnergyCheckpoint:
    """
    Callback para NetKet que guarda los pesos de la red correspondientes 
    a la iteración con la menor energía (Ground State aproximado).
    
    Equivalente a tu BestIterKeeper original.
    """
    def __init__(self, 
                 hamiltonian: Any, 
                 baseline: float = np.inf, 
                 save_path: Optional[pathlib.Path] = None,
                 stop_variance: bool = False):
        
        self.hamiltonian = hamiltonian
        self.baseline = baseline
        self.save_path = save_path
        self.stop_variance = stop_variance
        
        self.best_energy = np.inf
        self.best_state_params = None

    def __call__(self, step: int, log_data: dict, driver: Any) -> bool:
        """Función ejecutada por NetKet en cada paso del VMC."""
        vstate = driver.state
        
        # Obtenemos la energía calculada en este paso
        # Intentamos usar la cache de NetKet para no recalcular si el driver ya lo hizo
        stats = log_data.get("_cached_energy_stats", None)
        if stats is None:
            stats = vstate.expect(self.hamiltonian)
            
        energystep = np.real(stats.mean)

        # Si encontramos una mejor energía, guardamos una copia de los pesos
        if energystep < self.best_energy:
            self.best_energy = energystep
            self.best_state_params = flax.core.copy(driver.state.parameters)

            if self.save_path is not None:
                with open(self.save_path, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state.parameters))

        # Criterios de parada temprana (Early Stopping)
        if self.stop_variance:
            return True # Detiene la simulación si es True (ajustar lógica según necesidad)
            
        return True # Retornar True continúa la simulación


def build_observables_logger(metrics_history: Dict[str, List], 
                             hamiltonian: Any, 
                             wp_operators: Optional[List[Any]] = None):
    """
    Construye un callback que extrae la energía, varianza y observables adicionales 
    (como los bucles de Wilson Wp) en cada paso del entrenamiento.
    """
    def extract_metrics(step: int, log_data: dict, driver: Any) -> bool:
        vstate = driver.state
        
        # Extraer Energía
        stats = log_data.get("_cached_energy_stats", None)
        if stats is None:
            stats = vstate.expect(hamiltonian)

        energy = float(np.real(stats.mean))
        energy_error = float(np.real(stats.error_of_mean))
        
        # Extraer Varianza desde el loss_name de NetKet
        variance = float(np.real(getattr(log_data[driver._loss_name], "variance")))

        metrics_history['step'].append(step)
        metrics_history['energy'].append(energy)
        metrics_history['energy_error'].append(energy_error)
        metrics_history['variance'].append(variance)
        
        log_msg = f"Step {step:4d} | E = {energy:.6f} ± {energy_error:.1e} | Var = {variance:.4f}"

        # Extraer Bucles de Wilson (Plaquetas)
        if wp_operators is not None:
            wp_values = [float(np.real(vstate.expect(op).mean)) for op in wp_operators]
            wp_mean = np.mean(wp_values)
            metrics_history['wp_mean'].append(wp_mean)
            
            for idx, val in enumerate(wp_values):
                key = f'Wp_{idx}'
                if key not in metrics_history:
                    metrics_history[key] = []
                metrics_history[key].append(val)
                
            log_msg += f" | Wp_avg = {wp_mean:.4f}"

        # Imprimir en consola de forma limpia
        print(log_msg)
        return True

    return extract_metrics


class BestIterKeeper:
    """Almacena los valores de varias cantidades de la mejor iteración.

    "Mejor" se define en el sentido de menor energía.

    Argumentos:
        Hamiltoniano: Un array que contiene la matriz del Hamiltoniano.
        N: Número de espines en la cadena.
        baseline: Un límite inferior para la puntuación V. Si la puntuación V
            de la mejor iteración cae por debajo de este umbral, el proceso
            se detendrá antes.
        filename: Puede ser None o un archivo donde se escribirá el mejor estado.
    """

    def __init__(
        self,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float,
        filename: Optional[pathlib.Path] = None,
        stop_variance: bool = False
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.stop_variance = stop_variance
        self.vscore = np.inf
        self.best_energy = np.inf
        self.best_state = None

    def update(self, step, log_data, driver):
        """Actualiza las cantidades almacenadas si es necesario.

        Esta función está diseñada para actuar como una función de *callback* para NetKet.
        Por favor, consulta la documentación de su API para una explicación detallada.
        """

        vstate = driver.state
        energystep = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        varstep = self.N * var / mean**2

        if self.best_energy > energystep:
            self.best_energy = energystep
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(
                driver.state.parameters
            )
            self.vscore = varstep

            if self.filename != None:
                with open(self.filename, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state))
        if self.stop_variance==True:
           return True
        else:
            return self.vscore > self.baseline
        

class BestOverlapKeeper:
    """Almacena los valores de varias cantidades de la mejor iteración.

    "Mejor" se define en el sentido de menor energía.

    Argumentos:
        Hamiltoniano: Un array que contiene la matriz del Hamiltoniano.
        N: Número de espines en la cadena.
        baseline: Un límite inferior para la puntuación V. Si la puntuación V
            de la mejor iteración cae por debajo de este umbral, el proceso
            se detendrá antes.
        filename: Puede ser None o un archivo donde se escribirá el mejor estado.
    """

    def __init__(
        self,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float,
        filename: Optional[pathlib.Path] = None,
        stop_variance: bool = False
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.stop_variance = stop_variance
        self.vscore = np.inf
        self.best_energy = np.inf
        self.best_state = None
        self.best_infid = np.inf

    def update(self, step, log_data, driver):
        """Actualiza las cantidades almacenadas si es necesario.

        Esta función está diseñada para actuar como una función de *callback* para NetKet.
        Por favor, consulta la documentación de su API para una explicación detallada.
        """

        vstate = driver.state
        energystep = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        infidelity = np.inf
        if log_data['Infidelity'] is not None:
            infidelity = float(jnp.real(log_data['Infidelity'].mean))
        varstep = self.N * var / mean**2
        
        if self.best_infid > infidelity:
            self.best_infid = infidelity
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(
                driver.state.parameters
            )
            self.vscore = varstep

            if self.filename != None:
                with open(self.filename, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state))
        if self.stop_variance==True:
           return True
        else:
            return self.vscore > self.baseline
        
def make_extract_metrics(metrics_history, H):
  '''
  Function that extractus some metrics from the training proccess of the NQS.
  Please refer to NetKet documentation to learn more about the structure of this type of function.
  '''
  def extract_metrics(step, log_data, driver):
      stats = driver.state.expect(H)
      energy = float(jnp.real(stats.mean))
      energy_error = float(jnp.real(stats.error_of_mean))

      loss = float(jnp.real(getattr(log_data[driver._loss_name], "mean")))
      variance = float(jnp.real(getattr(log_data[driver._loss_name], "variance")))

      metrics_history['step'].append(step)
      metrics_history['energy'].append(energy)
      metrics_history['energy_error'].append(energy_error)
      #metrics_history['loss'].append(loss)
      metrics_history['variance'].append(variance)

      print(f"Step {step}: Energy = {energy:.6f} ± {energy_error:.2e}, Variance = {variance:.4f}")
      return True

  return extract_metrics


def make_extract_metrics_plaquete(metrics_history, H, Wp_op=None):
    def extract_metrics(step, log_data, driver):
        if log_data['Infidelity'] is not None:
            infidelity = float(jnp.real(log_data['Infidelity'].mean))
            metrics_history['infidelity'].append(infidelity)
        stats = log_data.get("_cached_energy_stats", None)
        if stats is None:
            stats = driver.state.expect(H)

        energy       = float(jnp.real(stats.mean))
        energy_error = float(jnp.real(stats.error_of_mean))
        variance     = float(jnp.real(getattr(log_data[driver._loss_name], "variance")))

        # --- CAMBIO AQUÍ: Calculamos cada Wp individualmente ---
        wp_values = [float(np.real(driver.state.expect(op).mean)) for op in Wp_op]
        wp_mean = np.mean(wp_values)

        # Guardar en el historial
        metrics_history['step'].append(step)
        metrics_history['energy'].append(energy)
        metrics_history['energy_error'].append(energy_error)
        metrics_history['variance'].append(variance)
        metrics_history['wp_mean'].append(wp_mean)

        # Guardamos cada plaqueta con una llave dinámica: Wp_0, Wp_1, ...
        for idx, val in enumerate(wp_values):
            key = f'Wp_{idx}'
            if key not in metrics_history:
                metrics_history[key] = []
            metrics_history[key].append(val)

        print(f"Step {step:4d} | E = {energy:.6f} | Wp_avg = {wp_mean:.4f}")
        return True

    return extract_metrics