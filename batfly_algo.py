import numpy as np

class BatFlyAlgorithm:
    """
    Classe para a meta-heurística do voo dos morcegos
    """
    
    def __init__(self, 
                 population_size: int, gene_length: int, f_min: float = 0.0, 
                 f_max: float = 2.0, alpha: float = 0.9, gamma: float = 0.9, 
                 loudness_init: float = 1.0, pulse_rate_init: float = 0.5):
        
        self.population_size = population_size
        self.gene_length = gene_length
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha
        self.gamma = gamma
        self.A = np.full(population_size, loudness_init)    # Loudness
        self.r = np.full(population_size, pulse_rate_init)  # Pulse rate
        self.frequencies = np.zeros(population_size)
        self.velocities = np.zeros((population_size, gene_length))
        self.population = self._inicialize_bat_population()
        self.best = self.population[0].copy()
    
    def _inicialize_bat_population(self) -> np.ndarray:
        return np.random.uniform(-1, 1, (self.population_size, self.gene_length))
    
    def _update_frequencies(self):
        beta = np.random.rand(self.population_size)
        self.frequencies = self.f_min + (self.f_max - self.f_min) * beta
    
    def _move_bats(self):
        self._update_frequencies()
        freq = self.frequencies[:, np.newaxis]  # shape (N, 1)
        self.velocities += (self.population - self.best) * freq
        self.population += self.velocities
    
    def _local_search(self, i: int) -> np.ndarray:
        epsilon = np.random.uniform(-1, 1, self.gene_length)
        return self.population[i] + epsilon * np.mean(self.A)
    
    def evolve(self, fitness_function, parallel=False) -> tuple[np.ndarray, float]:
        if parallel:
            fitness_scores = np.array(fitness_function(self.population))
        else:
            fitness_scores = np.array([fitness_function(ind) for ind in self.population])

        best_idx = np.argmax(fitness_scores)
        if best_idx >= 0:
            self.best = self.population[best_idx].copy()

        new_population = []

        for i in range(self.population_size):
            if np.random.rand() > self.r[i]:
                new_solution = self._local_search(i)
            else:
                new_solution = self.population[i].copy()

            new_fitness = fitness_function(np.array([new_solution]))[0]
            current_fitness = fitness_function(np.array([self.population[i]]))[0]

            if (np.random.rand() < self.A[i]) and (new_fitness > current_fitness):
                self.population[i] = new_solution
                self.A[i] *= self.alpha
                self.r[i] = self.r[i] * (1 - np.exp(-self.gamma))

            new_population.append(self.population[i])

        self.population = np.array(new_population)

        return self.best, fitness_function(self.best)

    def get_population(self) -> np.ndarray:
        return self.population

    def set_population(self, new_population: np.ndarray):
        self.population = new_population