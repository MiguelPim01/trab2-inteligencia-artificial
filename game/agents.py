import numpy as np
from abc import ABC, abstractmethod
from typing import List

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
    def __init__(self):
        #TODO
        pass
    
    def predict(self, state: np.ndarray) -> int:
        #TODO
        pass
