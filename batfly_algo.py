import numpy as np
import time

class BatFlyAlgorithm:
    """
    Classe para realizar a meta-heurística do voo dos morcegos
    """
    
    def __init__(self, 
                 population_size: int, 
                 gene_length: int, 
                 f_min: float = 0.0, 
                 f_max: float = 2.0, 
                 alpha: float = 0.9, 
                 gamma: float = 0.9):
        
        # Tamanhos e constantes
        self.population_size = population_size
        self.gene_length     = gene_length
        self.f_min           = f_min # 0
        self.f_max           = f_max # 2
        self.alpha           = alpha # 0.9 como define o artigo
        self.gamma           = gamma # 0.9 como define o artigo
        
        # Loudness e Pulse rate
        self.A = np.random.uniform(1.0, 2.0, population_size)  # Loudness inicial [1, 2]
        self.r = np.random.uniform(0.0, 1.0, population_size)  # Pulse rate inicial [0, 1]
        
        # Frequências iniciais
        self.frequencies = np.random.uniform(f_min, f_max, population_size)
        
        # Velocidades iniciais
        self.velocities = np.zeros((population_size, gene_length))
        
        # População inicial
        self.population = self._inicialize_bat_population()
        
        # Inicializando o melhor conjunto de pesos, a iteração e os valores iniciais de pulse rate
        self.best = self.population[0].copy()
        self.iteration = 0
        self.r0 = np.copy(self.r)
        self.A0 = np.copy(self.A)
    
    def _inicialize_bat_population(self) -> np.ndarray:
        return np.random.uniform(-1, 1, (self.population_size, self.gene_length))
    
    def _update_frequencies(self):
        """
        Faz o update das frequências.
        """
        
        # Gerando n valores beta aleatórios para atualizar a frequência de todos os morcegos
        beta = np.random.rand(self.population_size)
        
        # Atualizando as frequências
        self.frequencies = self.f_min + (self.f_max - self.f_min) * beta
    
    def _move_bats(self):
        """
        Faz o update das frequências, das velocidades e posições dos morcegos.
        """
        
        # Atualizando frequências (Equação 2)
        self._update_frequencies()
        freq = self.frequencies[:, np.newaxis]  # shape (N, 1)
        
        # Atualizando velocidade (Equação 3)
        self.velocities += (self.population - self.best) * freq
        
        # Atualizando posições dos morcegos (Equação 4)
        self.population += self.velocities
    
    def _local_search(self) -> np.ndarray:
        """
        Função responsável por gerar uma nova solução local próxima da melhor.

        Returns:
            new_solution (np.ndarray): Nova solução próxima a melhor.
        """
        # Gerando epsilon aleatório entre [-1, 1] para cada morcego
        epsilon = np.random.uniform(-1, 1, self.gene_length)
        
        # Nova solução próxima ao melhor
        new_solution = self.best + (epsilon*np.mean(self.A))
        
        return new_solution
    
    def evolve(self, fitness_function, parallel=False) -> tuple[np.ndarray, float]:
        # Movendo os morcegos
        self._move_bats()
        
        # Primeiro, iremos pegar os resultados de cada um dos individuos da população
        if parallel:
            fitness_scores = np.array(fitness_function(self.population))
        else:
            fitness_scores = np.array([fitness_function(ind) for ind in self.population])

        # Extraindo o indivíduo com o melhor desempenho médio da população
        best_idx = np.argmax(fitness_scores)
        self.best = self.population[best_idx].copy()    # Melhores pesos
        best_fitness = fitness_scores[best_idx]         # Melhor score

        # Atualizando população
        for i in range(self.population_size):
            
            # Obtendo a novo indivíduo da população
            if np.random.rand() > self.r[i]:
                new_solution = self._local_search()
                new_fitness = fitness_function(np.array([new_solution]))[0]
            else:
                new_solution = self.population[i].copy()
                new_fitness = fitness_scores[i]
            
            # Verificando se a nova solução será aceita ou não
            if (np.random.rand()*self.A0[i] < self.A[i]) and (new_fitness > fitness_scores[i]):
                self.population[i]  = new_solution
                self.A[i]          *= self.alpha
                self.r[i]           = self.r0[i] * (1 - np.exp(-self.gamma*self.iteration))
                
                if new_fitness > best_fitness:
                    self.best = new_solution.copy()
                    best_fitness = new_fitness

        return self.best, best_fitness
    
    def run_algorithm(self, game_fitness_function, max_iter : int = 1000, max_time : int = 12*60*60) -> tuple[np.ndarray, float, list]:
        # Inicializando variáveis
        best_weights_overall = None
        best_fitness_overall = -np.inf
        bests_scores = []
        start = time.time()

        for generation in range(max_iter):
            # Atualizando iteração e tempo
            self.iteration   = generation + 1
            start_generation = time.time()
            
            # Obtendo o melhor conjunto de pesos e o score relacionado a esse peso
            current_best_weights, current_best_fitness  = self.evolve(game_fitness_function, parallel=True)
            
            # Atualizando a lista de melhores scores a cada iteração
            bests_scores.append(current_best_fitness)

            if current_best_fitness > best_fitness_overall:
                # Atualizando novos melhores pesos
                best_fitness_overall = current_best_fitness
                best_weights_overall = current_best_weights
                print(f'Backup generation -> Melhor Fitness Geral: {best_fitness_overall.item():.2f}')
                
                # Salvando os melhores pesos
                np.save("best_weights.npy", best_weights_overall)
            
            end = time.time()
            print(f"[{generation + 1}/{max_iter}] Iterations --> Best Fitness: {current_best_fitness.item():.2f}, Melhor Fitness Geral: {best_fitness_overall.item():.2f} ({end-start_generation:.2f} s | tempo total: {(time.time()-start)/3600:.2f} h)")
            # print(f"Mean Loudness: {np.mean(self.A):.4f} | Mean frequencia: {np.mean(self.frequencies):.4f} | Mean velocidade: {np.mean(self.velocities):.4f}")
            
            # Verificando se chegou ao limite de tempo do algoritmo
            if time.time() - start > max_time:
                print("\n ### TIME OUT DE 12 HORAS - FINALIZANDO ITERAÇÕES DO ALGORITMO ###")
                break
        
        return best_weights_overall, best_fitness_overall, bests_scores
    
    def get_population(self) -> np.ndarray:
        return self.population

    def set_population(self, new_population: np.ndarray):
        self.population = new_population