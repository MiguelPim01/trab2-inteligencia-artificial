# Neural Network Training with Bat Algorithm

**Trabalho II da disciplina de Inteligência Artificial - UFES**

This project implements a metaheuristic optimization approach using the **Bat Algorithm** to train a neural network agent to play a survival obstacle avoidance game. The agent learns to navigate through increasingly difficult obstacles by optimizing its neural network weights through the bat-inspired optimization algorithm.

## 📋 Table of Contents

- [Overview](#overview)
- [Game Description](#game-description)
- [Neural Network Architecture](#neural-network-architecture)
- [Bat Algorithm Implementation](#bat-algorithm-implementation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)

## 🎯 Overview

The project combines:
- **Metaheuristic Optimization**: Bat Algorithm for neural network weight optimization
- **Neural Networks**: Multi-layer perceptron for game state prediction
- **Game Environment**: Custom survival game built with Pygame
- **Performance Analysis**: Statistical comparison with other agent types

The Bat Algorithm is a nature-inspired metaheuristic that mimics the echolocation behavior of bats for solving optimization problems. In this project, it optimizes 1,475 neural network weights to maximize the agent's survival score.

## 🎮 Game Description

The survival game features:
- **Objective**: Avoid moving obstacles for as long as possible
- **Actions**: Move up, move down, or stay in position
- **Sensor System**: 5×5 grid sensor that detects obstacles within a 250-pixel range
- **Difficulty Progression**: Obstacles speed increases over time
- **Scoring**: Continuous scoring based on survival time

### Game Mechanics
- Screen size: 800×600 pixels
- Player position: Fixed at x=100, movable along y-axis
- Obstacles: 30×30 pixel squares moving from right to left
- Obstacle speed: Starts at 7 px/frame and increases up to 28 px/frame
- Collision detection: Based on player radius (15 pixels)

## 🧠 Neural Network Architecture

The agent uses a feedforward neural network with the following structure:

```
Input Layer:    27 neurons (25 sensor grid + y-position + obstacle speed)
Hidden Layer 1: 32 neurons (tanh activation)
Hidden Layer 2: 16 neurons (tanh activation)
Output Layer:   3 neurons (softmax activation - action probabilities)
```

**Total Parameters**: 1,475 weights and biases
- Layer 1: 27×32 + 32 = 896
- Layer 2: 32×16 + 16 = 528
- Layer 3: 16×3 + 3 = 51

**Input Features**:
- 25 values: Flattened 5×5 sensor grid (binary obstacle detection)
- 1 value: Normalized player y-position (0-1)
- 1 value: Normalized current obstacle speed (0-1)

**Output Actions**:
- 0: No action (stay)
- 1: Move up
- 2: Move down

## 🦇 Bat Algorithm Implementation

### Parameters
- **Population Size**: 100 bats
- **Dimensions**: 1,475 (neural network weights)
- **Frequency Range**: [0.0, 3.0]
- **Loudness (A)**: [1.0, 2.0], decreases by factor α = 0.9
- **Pulse Rate (r)**: [0, 0.25], increases by factor γ = 0.9
- **Max Iterations**: 1,000
- **Fitness Evaluation**: Average score over 3 game runs per agent

### Algorithm Flow
1. **Initialization**: Random population of weight vectors in range [-2, 2]
2. **Fitness Evaluation**: Each bat plays the game 3 times, fitness = average score
3. **Solution Search**:
   - With probability r: Generate local solution near best solution
   - Otherwise: Update velocity and position (bat "flies")
4. **Selection**: Accept new solution if better and random < loudness A
5. **Parameter Update**: Decrease loudness, increase pulse rate
6. **Iteration**: Repeat until convergence or max iterations

### Key Features
- **Parallel Evaluation**: Multiple agents play simultaneously for efficiency
- **Adaptive Parameters**: Loudness and pulse rate adjust during optimization
- **Local vs Global Search**: Balanced exploration and exploitation

## 📁 Project Structure

```
trab2-inteligencia-artificial/
├── batfly_algo.py              # Bat Algorithm implementation
├── main.py                     # Training script and entry point
├── test_trained_agent.py       # Testing trained agents
├── Trab2_Miguel_Pim.ipynb     # Results analysis notebook
├── requirements.txt            # Project dependencies
├── best_weights.npy           # Trained neural network weights
├── best_scores_per_iter.npy   # Training history
├── game/
│   ├── agents.py              # Agent implementations (NN, Human, Rule-based)
│   ├── config.py              # Game configuration
│   ├── core.py                # Game engine and mechanics
├── docs/
│   └── bat.pdf                # Bat Algorithm reference paper
├── figs/                      # Generated plots and visualizations
└── backup/                    # Backup of best results
```

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/MiguelPim01/trab2-inteligencia-artificial.git
cd trab2-inteligencia-artificial
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- numpy >= 2.3.2
- pygame >= 2.6.1

## 🚀 Usage

### Training a New Agent

To train a new neural network agent using the Bat Algorithm:

```bash
python main.py
```

This will:
- Initialize a population of 100 bats (neural networks)
- Run up to 1,000 iterations of optimization
- Save the best weights to `best_weights.npy`
- Save training history to `best_scores_per_iter.npy`
- Generate a training curve plot in `figs/scores_curve.png`
- Test the final agent over 30 game runs

**Training Parameters** (configurable in `main.py`):
- `POPULATION_SIZE`: Number of bats (default: 100)
- `MAX_ITER`: Maximum iterations (default: 1,000)
- `NUM_GAMES`: Games per fitness evaluation (default: 3)
- `FPS`: Game speed (default: 1,000 for fast training)
- `RENDER`: Enable/disable game visualization (default: False)

### Testing a Trained Agent

To test an existing trained agent:

```bash
python test_trained_agent.py
```

Or use it as a module:
```python
from test_trained_agent import test_agent
import numpy as np

weights = np.load("best_weights.npy")
test_agent(weights, num_tests=30, render=True)
```

### Analyzing Results

Open the Jupyter notebook for detailed analysis:

```bash
jupyter notebook Trab2_Miguel_Pim.ipynb
```

The notebook includes:
- Performance comparison with rule-based, human, and other neural agents
- Statistical tests (t-test, Wilcoxon)
- Box plots and visualizations
- Mean and standard deviation calculations

## 📊 Results

### Performance Comparison (30 test runs each)

| Agent Type       | Mean Score | Std Dev |
|-----------------|------------|---------|
| Rule-Based      | 13.97      | 5.67    |
| Human Player    | 22.19      | 8.11    |
| Neural Agent    | 38.99      | 17.01   |
| **Bat Algorithm** | **44.78** | **29.05** |

The Bat Algorithm successfully trains neural network agents that outperform both rule-based systems and human players on average, demonstrating the effectiveness of metaheuristic optimization for game-playing agents.

### Training Characteristics
- **Convergence**: Typically shows improvement over first 100-200 iterations
- **Best Scores**: Achieves scores of 30-70+ depending on training run
- **Training Time**: ~8-12 hours for 1,000 iterations (hardware dependent)
- **Stability**: Algorithm maintains best solution while exploring new areas

## 📦 Dependencies

- **numpy**: Numerical computations and array operations
- **pygame**: Game rendering and environment
- **pandas**: Data analysis (notebook)
- **scipy**: Statistical tests (notebook)
- **matplotlib**: Plotting and visualization (notebook)
- **seaborn**: Statistical visualizations (notebook)

## 🎓 Academic Context

This project was developed as part of the Artificial Intelligence course at the Federal University of Espírito Santo (UFES). It demonstrates practical applications of:
- Metaheuristic optimization algorithms
- Neural network training without backpropagation
- Evolutionary computation approaches
- Game AI development

## 📄 References

The Bat Algorithm implementation is based on the original paper by Xin-She Yang (2010). See `docs/bat.pdf` for theoretical background.

## 👤 Author

[Miguel Vieira Machado Pim](https://github.com/MiguelPim01) 
Federal University of Espírito Santo (UFES)

## 📝 License

See [LICENSE](LICENSE) file for details.
