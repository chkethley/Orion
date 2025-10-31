"""Neural Architecture Search (NAS) for automated model design."""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import random
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """Define the search space for NAS."""
    operations: List[str]
    num_layers_range: Tuple[int, int]
    hidden_dims: List[int]
    activation_functions: List[str]
    dropout_rates: List[float]
    use_batch_norm: bool = True
    use_residual: bool = True


class SearchableBlock(nn.Module):
    """A searchable block with multiple operation choices."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        operations: List[str]
    ):
        """Initialize searchable block."""
        super().__init__()

        self.operations = nn.ModuleDict()

        for op_name in operations:
            if op_name == 'linear':
                self.operations[op_name] = nn.Linear(input_dim, output_dim)
            elif op_name == 'conv1d':
                # For sequence data
                self.operations[op_name] = nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1)
            elif op_name == 'identity':
                if input_dim == output_dim:
                    self.operations[op_name] = nn.Identity()
                else:
                    self.operations[op_name] = nn.Linear(input_dim, output_dim)
            elif op_name == 'zero':
                self.operations[op_name] = lambda x: torch.zeros_like(x) if x.shape[-1] == output_dim else torch.zeros(*x.shape[:-1], output_dim, device=x.device)

        # Architecture parameters (weights for each operation)
        self.arch_params = nn.Parameter(torch.randn(len(operations)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted operations."""
        weights = torch.softmax(self.arch_params, dim=0)

        output = 0
        for i, (op_name, op) in enumerate(self.operations.items()):
            if op_name != 'zero':
                output = output + weights[i] * op(x)

        return output

    def get_selected_operation(self) -> str:
        """Get the operation with highest weight."""
        weights = torch.softmax(self.arch_params, dim=0)
        selected_idx = torch.argmax(weights).item()
        return list(self.operations.keys())[selected_idx]


class DARTS(nn.Module):
    """Differentiable Architecture Search (DARTS)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_cells: int,
        operations: List[str],
        hidden_dim: int = 128
    ):
        """Initialize DARTS model."""
        super().__init__()

        self.num_cells = num_cells

        # Create searchable cells
        self.cells = nn.ModuleList()
        prev_dim = input_dim

        for i in range(num_cells):
            cell_output_dim = hidden_dim if i < num_cells - 1 else output_dim
            cell = SearchableBlock(prev_dim, cell_output_dim, operations)
            self.cells.append(cell)
            prev_dim = cell_output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through searchable architecture."""
        for cell in self.cells:
            x = cell(x)
            x = torch.relu(x)

        return x

    def get_architecture(self) -> List[str]:
        """Get the discovered architecture."""
        return [cell.get_selected_operation() for cell in self.cells]

    def arch_parameters(self) -> List[nn.Parameter]:
        """Get architecture parameters for optimization."""
        return [cell.arch_params for cell in self.cells]


class EvolutionaryNAS:
    """Evolutionary algorithm for Neural Architecture Search."""

    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 20,
        num_generations: int = 50,
        mutation_rate: float = 0.2
    ):
        """Initialize evolutionary NAS."""
        self.search_space = search_space
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

        self.population: List[Dict[str, Any]] = []
        self.fitness_scores: List[float] = []

    def create_random_architecture(self) -> Dict[str, Any]:
        """Create a random architecture from search space."""
        num_layers = random.randint(*self.search_space.num_layers_range)

        architecture = {
            'num_layers': num_layers,
            'layers': []
        }

        for i in range(num_layers):
            layer_config = {
                'type': random.choice(self.search_space.operations),
                'hidden_dim': random.choice(self.search_space.hidden_dims),
                'activation': random.choice(self.search_space.activation_functions),
                'dropout': random.choice(self.search_space.dropout_rates),
                'batch_norm': self.search_space.use_batch_norm if random.random() > 0.5 else False,
                'residual': self.search_space.use_residual if random.random() > 0.5 else False
            }
            architecture['layers'].append(layer_config)

        return architecture

    def initialize_population(self):
        """Initialize population with random architectures."""
        self.population = [
            self.create_random_architecture()
            for _ in range(self.population_size)
        ]
        self.fitness_scores = [0.0] * self.population_size

    def mutate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an architecture."""
        mutated = architecture.copy()
        mutated['layers'] = [layer.copy() for layer in architecture['layers']]

        # Random mutations
        if random.random() < self.mutation_rate:
            # Add or remove layer
            if random.random() > 0.5 and mutated['num_layers'] < self.search_space.num_layers_range[1]:
                # Add layer
                new_layer = {
                    'type': random.choice(self.search_space.operations),
                    'hidden_dim': random.choice(self.search_space.hidden_dims),
                    'activation': random.choice(self.search_space.activation_functions),
                    'dropout': random.choice(self.search_space.dropout_rates),
                    'batch_norm': random.random() > 0.5,
                    'residual': random.random() > 0.5
                }
                insert_pos = random.randint(0, len(mutated['layers']))
                mutated['layers'].insert(insert_pos, new_layer)
                mutated['num_layers'] += 1
            elif mutated['num_layers'] > self.search_space.num_layers_range[0]:
                # Remove layer
                remove_pos = random.randint(0, len(mutated['layers']) - 1)
                mutated['layers'].pop(remove_pos)
                mutated['num_layers'] -= 1

        # Mutate individual layers
        for layer in mutated['layers']:
            if random.random() < self.mutation_rate:
                if random.random() > 0.7:
                    layer['hidden_dim'] = random.choice(self.search_space.hidden_dims)
                if random.random() > 0.7:
                    layer['activation'] = random.choice(self.search_space.activation_functions)
                if random.random() > 0.7:
                    layer['dropout'] = random.choice(self.search_space.dropout_rates)

        return mutated

    def crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover two parent architectures."""
        # Simple crossover: take layers from both parents
        crossover_point = min(parent1['num_layers'], parent2['num_layers']) // 2

        child = {
            'num_layers': crossover_point + (parent2['num_layers'] - crossover_point),
            'layers': (
                parent1['layers'][:crossover_point] +
                parent2['layers'][crossover_point:]
            )
        }

        return child

    def select_parents(self, num_parents: int) -> List[Dict[str, Any]]:
        """Select parents using tournament selection."""
        parents = []
        for _ in range(num_parents):
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])

        return parents

    def evolve(self) -> List[Dict[str, Any]]:
        """Run evolutionary search."""
        logger.info("Starting evolutionary NAS")

        # Initialize
        self.initialize_population()

        for generation in range(self.num_generations):
            logger.info(f"Generation {generation + 1}/{self.num_generations}")

            # Selection
            num_parents = self.population_size // 2
            parents = self.select_parents(num_parents)

            # Generate offspring
            offspring = []

            while len(offspring) < self.population_size - num_parents:
                # Crossover
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child)

                offspring.append(child)

            # New population: elites + offspring
            # Sort by fitness and keep top performers
            sorted_indices = np.argsort(self.fitness_scores)[::-1]
            elites = [self.population[i] for i in sorted_indices[:num_parents]]

            self.population = elites + offspring
            self.fitness_scores = [0.0] * self.population_size

            # Note: Fitness evaluation happens externally

        # Return best architecture
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]


class RandomSearch:
    """Simple random search for architecture."""

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 100
    ):
        """Initialize random search."""
        self.search_space = search_space
        self.num_samples = num_samples

    def search(self) -> List[Dict[str, Any]]:
        """Generate random architectures."""
        architectures = []

        for _ in range(self.num_samples):
            num_layers = random.randint(*self.search_space.num_layers_range)

            arch = {'num_layers': num_layers, 'layers': []}

            for _ in range(num_layers):
                layer = {
                    'type': random.choice(self.search_space.operations),
                    'hidden_dim': random.choice(self.search_space.hidden_dims),
                    'activation': random.choice(self.search_space.activation_functions),
                    'dropout': random.choice(self.search_space.dropout_rates),
                }
                arch['layers'].append(layer)

            architectures.append(arch)

        return architectures


def build_model_from_architecture(
    architecture: Dict[str, Any],
    input_dim: int,
    output_dim: int
) -> nn.Module:
    """Build a PyTorch model from architecture specification."""
    layers = []
    prev_dim = input_dim

    for i, layer_config in enumerate(architecture['layers']):
        hidden_dim = layer_config['hidden_dim']

        # Main layer
        if layer_config['type'] == 'linear':
            layers.append(nn.Linear(prev_dim, hidden_dim))
        elif layer_config['type'] == 'conv1d':
            layers.append(nn.Conv1d(prev_dim, hidden_dim, kernel_size=3, padding=1))

        # Batch norm
        if layer_config.get('batch_norm', False):
            layers.append(nn.BatchNorm1d(hidden_dim))

        # Activation
        activation = layer_config.get('activation', 'relu')
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'silu':
            layers.append(nn.SiLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Dropout
        dropout = layer_config.get('dropout', 0.0)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        prev_dim = hidden_dim

    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


class HyperparameterOptimizer:
    """Bayesian optimization for hyperparameter tuning."""

    def __init__(
        self,
        param_space: Dict[str, List[Any]],
        num_iterations: int = 50
    ):
        """Initialize hyperparameter optimizer."""
        self.param_space = param_space
        self.num_iterations = num_iterations
        self.history: List[Tuple[Dict[str, Any], float]] = []

    def suggest_params(self) -> Dict[str, Any]:
        """Suggest next set of hyperparameters."""
        if len(self.history) < 10:
            # Random search initially
            params = {
                key: random.choice(values)
                for key, values in self.param_space.items()
            }
        else:
            # Use history to guide search (simplified)
            # In practice, use proper Bayesian optimization
            best_params = max(self.history, key=lambda x: x[1])[0]

            params = best_params.copy()
            # Perturb one parameter
            key_to_change = random.choice(list(self.param_space.keys()))
            params[key_to_change] = random.choice(self.param_space[key_to_change])

        return params

    def report_result(self, params: Dict[str, Any], score: float):
        """Report evaluation result."""
        self.history.append((params, score))
        logger.info(f"Evaluated params with score {score:.4f}")

    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters found."""
        if not self.history:
            return {}

        return max(self.history, key=lambda x: x[1])[0]
