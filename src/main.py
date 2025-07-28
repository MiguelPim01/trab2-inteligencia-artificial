from game.config import GameConfig
from game.core import SurvivalGame
from game.agents import NeuralNetworkAgent

import numpy as np
import time

NUM_PLAYERS      = 30            # Número de agentes que irão jogar o jogo
NUM_PLAYED_GAMES = 3             # Quantidade de vezes que cada agente irá jogar o jogo
MAX_TIME         = 12 * 60 * 60  # 12 horas em segundos
MAX_ITER         = 1000          # Máximo de iterações do algoritmo

def main():
    total_scores = np.zeros(NUM_PLAYERS)
    
    for _ in range(NUM_PLAYED_GAMES):
        # Looping principal do jogo
        game_config = GameConfig(num_players=NUM_PLAYERS)     # Configurações do jogo
        game = SurvivalGame(config=game_config, render=False) # Criando o jogo
        
        # Inicializando variáveis para o jogo
        iterations = 0
        start = time.process_time()                                 # Definindo o início do cronometro
        agents = [NeuralNetworkAgent() for _ in range(NUM_PLAYERS)] # Criando 30 agentes para jogar o jogo
        
        while not game.all_players_dead() and iterations <= MAX_ITER and time.process_time()-start <= MAX_TIME:
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
            
            # Incrementando o número de iterações
            iterations += 1
        
        for i, player in enumerate(game.players):
            total_scores[i] += player.score
    
    avg_scores = total_scores/3

if __name__ == "__main__":
    main()