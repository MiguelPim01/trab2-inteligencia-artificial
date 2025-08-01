import numpy as np
from abc import ABC, abstractmethod
from typing import List
from game.core import GameConfig

class Agent(ABC):
    """
    Interface para todos os agentes.
    """
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """
        Faz uma previsão de ação com base no estado atual.
        """
        pass

class HumanAgent(Agent):
    """
    Agente controlado por um humano (para modo manual).
    """
    def predict(self, state: np.ndarray) -> int:
        # O estado é ignorado - entrada vem do teclado
        return 0  # Padrão: não fazer nada (será sobrescrito pela entrada do usuário no manual_play.py)

class NeuralNetworkAgent(Agent):
    def __init__(self, config: GameConfig, gene_vector: np.ndarray):
        self.config = config
        self.input_size = config.sensor_grid_size * config.sensor_grid_size + 2  # incluindo variáveis internas
        self.hidden1_size = 32
        self.hidden2_size = 16
        self.output_size = 3  # ações: noop, cima, baixo

        # Descompacta o vetor de genes em pesos e bias
        self._unpack_genes(gene_vector)

    def _unpack_genes(self, gene_vector: np.ndarray):
        # Cálculo do número de parâmetros esperados
        idx = 0

        def extract_weights(in_dim, out_dim):
            nonlocal idx
            w_size = in_dim * out_dim
            b_size = out_dim
            w = gene_vector[idx : idx + w_size].reshape((in_dim, out_dim))
            idx += w_size
            b = gene_vector[idx : idx + b_size]
            idx += b_size
            return w, b

        self.w1, self.b1 = extract_weights(self.input_size, self.hidden1_size)
        self.w2, self.b2 = extract_weights(self.hidden1_size, self.hidden2_size)
        self.w3, self.b3 = extract_weights(self.hidden2_size, self.output_size)

    def predict(self, state: np.ndarray) -> int:
        # Rede neural com 3 camadas totalmente conectadas
        def tanh(x):
            return np.tanh(x)

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / np.sum(e_x)

        x = state  # entrada do estado já como vetor 1D

        # Forward pass
        h1 = tanh(np.dot(x, self.w1) + self.b1)
        h2 = tanh(np.dot(h1, self.w2) + self.b2)
        out = softmax(np.dot(h2, self.w3) + self.b3)

        action = int(np.argmax(out))
        return action
