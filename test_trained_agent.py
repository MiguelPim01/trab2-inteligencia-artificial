import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent
import os

def test_agent(weights: np.ndarray, num_tests: int = 30, render: bool = False):
    print(f"\n--- Testando Agente Treinado por {num_tests} vezes ---")
    
    # Configurações do jogo para o teste
    game_config = GameConfig(render_grid=True)

    total_scores = []

    for i in range(num_tests):
        game = SurvivalGame(config=game_config, render=render)
        agent = NeuralNetworkAgent(gene_vector=weights)
        
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            
            game.update([action])
            if render:
                game.render_frame()
        
        final_score = game.players[0].score
        total_scores.append(final_score)
        print(f"Teste {i+1}/{num_tests}: Score Final = {final_score:.2f}")

    avg_score = np.mean(total_scores)
    std_score = np.std(total_scores)
    print(f"\nResultados Finais após {num_tests} testes:")
    print(f"Score Médio: {avg_score.item():.2f}")
    print(f"Desvio Padrão do Score: {std_score.item():.2f}")

if __name__ == "__main__":
    best_trained_weights = np.load('best_weights.npy') 

    test_agent(best_trained_weights, num_tests=30, render=False)

