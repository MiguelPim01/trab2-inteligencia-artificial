from game.config import GameConfig
from game.core import SurvivalGame
from game.agents import NeuralNetworkAgent
from batfly_algo import BatFlyAlgorithm
from test_trained_agent import test_agent

import numpy as np
import time
import os

import matplotlib.pyplot as plt

NUM_PLAYED_GAMES = 3             # Quantidade de vezes que cada agente irá jogar o jogo
MAX_TIME         = 12 * 60 * 60  # 12 horas em segundos
MAX_ITER         = 1000          # Máximo de iterações do algoritmo
FPS              = 1000          # Quantidade de fps do jogo
RENDER           = False

POPULATION_SIZE = 100

INPUT_SIZE = 27
HIDDEN1_SIZE = 32
HIDDEN2_SIZE = 16
OUTPUT_SIZE = 3

GENE_LENGTH = INPUT_SIZE*HIDDEN1_SIZE + HIDDEN1_SIZE + HIDDEN1_SIZE*HIDDEN2_SIZE + HIDDEN2_SIZE + HIDDEN2_SIZE*OUTPUT_SIZE + OUTPUT_SIZE  # = 1475

def game_fitness_function(population : np.ndarray) -> np.ndarray:
    if population.ndim == 1:
        population = np.expand_dims(population, axis=0)
    
    game_config = GameConfig(num_players=len(population), fps=FPS)
    agents = [NeuralNetworkAgent(gene_vector=individual) for individual in population]  # Criando 30 agentes para jogar o jogo
    
    total_scores = np.zeros(len(agents)) # Array com os scores de cada agente
    
    for _ in range(NUM_PLAYED_GAMES):
        # Criando o jogo
        game = SurvivalGame(config=game_config, render=RENDER)
        
        while not game.all_players_dead():
            # Próximas ações que cada um dos agentes irá tomar
            actions = []
            
            for i, agent in enumerate(agents):
                if game.players[i].alive:
                    cur_state = game.get_state(i, include_internals=True) # Obtendo estado atual do jogador
                    action = agent.predict(state=cur_state)               # Prevendo ação para chegar no próximo melhor estado do jogador
                    actions.append(action)                                # Adicionando a ação para chegar ao próximo estado do jogador atual
                else:
                    actions.append(0) # Nenhuma ação deve ser feita se o jogador está morto
            
            # Cada agente deve realizar a próxima ação
            game.update(actions=actions)
            
            if game.render:
                game.render_frame() # Se o jogo tiver renderização ele renderiza na tela
        
        for i, player in enumerate(game.players):
            total_scores[i] += player.score
    
    avg_scores = total_scores/3
    
    return avg_scores

def plot_scores_curve(best_scores):
    os.makedirs("figs", exist_ok=True)
    
    iterations = [i+1 for i in range(len(best_scores))]
    
    plt.plot(iterations, best_scores)
    plt.xlabel("Iteração")
    plt.ylabel("Pontuação")
    plt.title("Melhores Pontuações dos Agentes vs Iterações do Algoritmo")
    
    plt.savefig(os.path.join("figs", "scores_curve.png"), format="png")

def main():
    print("\n--- Iniciando Treinamento com Algoritmo Voo dos Morcegos ---")
    
    bfa = BatFlyAlgorithm(
        population_size=POPULATION_SIZE,
        gene_length=GENE_LENGTH
    )
    
    best_weights_overall, best_fitness_overall, bests_scores = bfa.run_algorithm(game_fitness_function=game_fitness_function)
    
    print("\n--- Treinamento Concluído ---")
    print(f"Melhor Fitness Geral Alcançado: {best_fitness_overall.item():.2f}")

    if best_weights_overall is not None:
        np.save("best_weights.npy", best_weights_overall)
        print("Melhores pesos salvos em \'best_weights.npy\'")
        
        test_agent(weights=best_weights_overall)
        plot_scores_curve(best_scores=bests_scores)
        
        np.save("best_scores_per_iter.npy", np.array(bests_scores))
        print("Melhores scores por iteração salvos em \'best_scores_per_iter.npy\'")
    else:
        print("Nenhum peso ótimo encontrado.")

if __name__ == "__main__":
    main()