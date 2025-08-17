import numpy as np
import time

class BatFlyAlgorithm:
    """
    Classe para realizar a meta-heurística do voo dos morcegos
    """
    
    def __init__(self, population_size: int, d: int):
        """Inicializa o algoritmo de voo dos morcegos.

        Args:
            population_size (int): Tamanho da população de morcegos.
            d (int): Dimensões dos atributos dos morcegos.
        """
        self.population_size = population_size # Quantidade de morcegos
        self.d = d # Tamanho do array de pesos
        
        self.solutions = np.random.uniform(-2, 2, size=(population_size, d)) # Criando vetor de soluções (100, 1475)
        self.bats = np.zeros((self.population_size, self.d)) # Criando população de morcegos (100, 1475)
        self.v = np.zeros((self.population_size, self.d)) # Inicialização das velocidades dos morcegos (100, 1475)
        
        self.best_solution = self.solutions[0].copy() # Inicializando a melhor solução
        
        self.f_min = 0.0 # Frequência mínima dos morcegos
        self.f_max = 3.0 # Frequência máxima dos morcegos
        self.freq = np.random.uniform(self.f_min, self.f_max, self.population_size) # Inicialização da frequência de cada morcego
        
        self.alpha = 0.9 # Fator de decremento da loudness
        self.gamma = 0.9 # Fator de incremento do pulse rate
        
        self.r_max = 1 # Pulse rate máximo
        self.r = np.random.uniform(0, self.r_max/4, self.population_size) # Inicialização do vetor de pulse rates
        self.A = np.random.uniform(1, 2, self.population_size) # Inicialização do vetor de loudness
        
        self._fitness_cache = None # Cache para guardar os scores das soluções
    
    def get_best_solution(self, objective_function, parallel : bool = True) -> float:
        """Obtém os scores de cada um dos morcegos no jogo e seleciona o melhor.

        Args:
            objective_function (_type_): Função objetivo.
            parallel (bool, optional): Defini se irá rodar o jogo em paralelo ou não. Defaults to True.

        Returns:
            float: Valor do melhor score obtido nessa rodada.
        """
        if parallel:
            scores = np.array(objective_function(self.solutions))
        else:
            scores = np.array([objective_function(weights) for weights in self.solutions])
        self._fitness_cache = scores
        
        best_indx = np.argmax(scores)
        self.best_solution = self.solutions[best_indx].copy()
        
        return scores[best_indx]
    
    def search_solutions(self, objective_function, iteration : int):
        """Realiza a busca por mais soluções voando com os morcegos ou gerando soluções locais próximas a melhor solução da rodada anterior.

        Args:
            objective_function (_type_): Função objetivo.
            iteration (int): Iteração atual do algoritmo.
        """
        # Preenchendo vetor de fitness cache apenas na primeira iteração
        if self._fitness_cache is None:
            self._fitness_cache = np.array(objective_function(self.solutions))
        
        # Atualiza as frequências
        self.freq = np.random.uniform(self.f_min, self.f_max, self.population_size)
        
        for i in range(self.population_size):
            if np.random.uniform(0, 1) < self.r[i]:
                # Gera uma solução próxima a melhor solução
                epsilon = np.random.uniform(-1, 1, self.d)
                self.bats[i] = self.best_solution + np.mean(self.A) * epsilon
            else:
                # Voa com o morcego
                self.v[i] = self.v[i] + (self.solutions[i] - self.best_solution) * self.freq[i]
                self.bats[i] = self.solutions[i] + self.v[i]
            
            # Obtendo scores para comparar
            bat_score = float(objective_function(self.bats[i]))
            cur_score = float(self._fitness_cache[i])
            
            if bat_score > cur_score:
                if np.random.uniform(0, 1) < self.A[i]:
                    # Solução aceita
                    self.solutions[i] = self.bats[i].copy()
                    self._fitness_cache[i] = bat_score
                    
                    # Atualizando loudness e pulse rate
                    self.A[i] *= self.alpha
                    self.r[i] = self.r_max * (1 - np.exp(-self.gamma * iteration))
                    
    def execute_algorithm(self, objective_function, total_iterations: int):
        """Executa o algoritmo de voo dos morcegos e retorna a melhor solução e o histórico de melhores scores.

        Args:
            objective_function (_type_): Função objetivo.
            total_iterations (int): Número máximo de iterações do algoritmo.

        Returns:
            best_weights (ndarray): Melhores pesos encontrados pelo algoritmo.
            history (list): Histórico de melhores scores por iteração do algoritmo.
        """
        # Iniciando cronômetro do algoritmo
        t_start = time.time()
        MAX_ELAPSED_TIME = 12*60*60
        
        best_score = 0
        
        # Histórico dos melhores scores por iteração
        history = []
        
        for i in range(total_iterations):
            iteration_start = time.time()
            
            self.search_solutions(objective_function=objective_function, iteration=i+1)
            best_iteration_score = self.get_best_solution(objective_function=objective_function, parallel=True)
            
            history.append(best_iteration_score)
            
            if best_iteration_score > best_score:
                # Atualizando o melhor score
                best_score = best_iteration_score
                print(f"Novo Melhor Encontrado --> {best_score:.2f}")
                
                # Salvando os melhores pesos
                np.save("best_weights.npy", self.best_solution)

            iteration_end = time.time()
            t_elapsed = time.time() - t_start
            
            print(f"[{i+1}/{total_iterations}] Best Iteration score: {best_iteration_score:.2f} | Best Score: {best_score:.2f} (Iteration time: {iteration_end-iteration_start:.2f} s -- Total time: {t_elapsed/3600:.2f} h)")
            
            if t_elapsed >= MAX_ELAPSED_TIME:
                print("TIMEOUT")
                break
    
        return self.best_solution, history