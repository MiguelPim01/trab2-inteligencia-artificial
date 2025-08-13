from game.config import GameConfig
from game.core import SurvivalGame
from game.agents import NeuralNetworkAgent
from batfly_algo import BatFlyAlgorithm
from test_trained_agent import test_agent

import numpy as np
import time
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt

NUM_GAMES = 3             # Quantidade de vezes que cada agente irá jogar o jogo
MAX_ITER         = 1000          # Máximo de iterações do algoritmo
FPS              = 1000          # Quantidade de fps do jogo
RENDER           = False

POPULATION_SIZE = 100

INPUT_SIZE = 27
HIDDEN1_SIZE = 32
HIDDEN2_SIZE = 16
OUTPUT_SIZE = 3

DIMESION = INPUT_SIZE*HIDDEN1_SIZE + HIDDEN1_SIZE + HIDDEN1_SIZE*HIDDEN2_SIZE + HIDDEN2_SIZE + HIDDEN2_SIZE*OUTPUT_SIZE + OUTPUT_SIZE  # = 1475

def game_fitness_function(population : np.ndarray) -> np.ndarray:
    """Joga o jogo 3 vezes para cada individuo da população

    Args:
        population (np.ndarray): Matriz de pesos

    Returns:
        avg_scores (np.ndarray): Pontuação média de cada agente para aquele conjunto de pesos
    """
    if population.ndim == 1:
        population = np.expand_dims(population, axis=0)
    
    game_config = GameConfig(num_players=len(population), fps=FPS)
    agents = [NeuralNetworkAgent(weight_vector=individual) for individual in population]  # Criando 30 agentes para jogar o jogo
    
    total_scores = np.zeros(len(agents)) # Array com os scores de cada agente
    
    for _ in range(NUM_GAMES):
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
    
    avg_scores = total_scores/NUM_GAMES
    
    return avg_scores

def plot_scores_curve(best_scores : list):
    """Cria um plot dos melhores scores alcançados vs iteração do algoritmo

    Args:
        best_scores (list): Vetor de scores do algoritmo
    """
    os.makedirs("figs", exist_ok=True)
    
    iterations = [i+1 for i in range(len(best_scores))]
    
    plt.plot(iterations, best_scores)
    plt.xlabel("Iteração")
    plt.ylabel("Pontuação")
    plt.title("Melhores Pontuações dos Agentes vs Iterações do Algoritmo")
    
    plt.savefig(os.path.join("figs", "scores_curve.png"), format="png")
    plt.close()

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument(
        "--train-network", '-tn',
        action='store_true',
        help="Defines wether the algorithm will run or not"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.train_network:
        print("\n--- Iniciando Treinamento com Voo dos Morcegos ---")
        
        bat_algorithm = BatFlyAlgorithm(
            population_size=POPULATION_SIZE,
            d=DIMESION
        )
        
        best_weights, scores_history = bat_algorithm.execute_algorithm(objective_function=game_fitness_function, total_iterations=MAX_ITER)
        
        print("--- Treinamento finalizado ---")
        
        np.save("best_weights.npy", best_weights)
        print("Melhores pesos salvos em \'best_weights.npy\'")
        
        np.save("best_scores_per_iter.npy", np.array(scores_history))
        print("Histórico de melhores scores salvos em \'best_scores_per_iter.npy\'")
    else:
        print("Para realizar o treinamento rode: \'python main.py -tn\'")
    
    # Testando o agente
    best_weights = np.load("best_weights.npy")
    test_agent(weights=best_weights, num_tests=30, render=RENDER)
    
    # Plotando curva de scores vs iteração
    scores_history = np.load("best_scores_per_iter.npy")
    plot_scores_curve(best_scores=scores_history)

if __name__ == "__main__":
    main()