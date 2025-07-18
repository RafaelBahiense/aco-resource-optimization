import random
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional, cast, overload


@overload
def f(x: float) -> float: ...
@overload
def f(x: np.ndarray) -> np.ndarray: ...
def f(x):
    """Calcula a função objetivo f(x) = x^2."""
    return x**2


class Ant:
    """Representa uma formiga no algoritmo ACO."""

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        """Inicializa a formiga com posição aleatória e avalia f(x)."""
        self.position: float = random.uniform(
            lower_bound, upper_bound
        )  # Posição inicial aleatória
        self.fitness: float = f(self.position)  # Avalia f(x) na posição inicial

    def move(
        self, pheromone_map: np.ndarray, lower_bound: float, upper_bound: float
    ) -> None:
        """Move a formiga dando um passo pequeno aleatório, respeitando os limites."""
        # Escolhe um índice ponderado pelo mapa de feromônio
        probs = pheromone_map / pheromone_map.sum()
        idx = np.random.choice(len(pheromone_map), p=probs)
        pos = lower_bound + idx * (upper_bound - lower_bound) / (len(pheromone_map) - 1)
        # Pequeno ruído local
        pos += random.uniform(-0.05, 0.05)
        pos = max(lower_bound, min(pos, upper_bound))
        self.position = pos
        self.fitness = f(self.position)


class ACO:
    """Algoritmo Ant Colony Optimization para minimizar f(x) = x^2."""

    def __init__(
        self,
        n_ants: int,
        n_iterations: int,
        lower_bound: float,
        upper_bound: float,
        evaporation_rate: float = 0.9,
        verbose: bool = False,
        early_stop_threshold: float = 1e-5,
        resolution: int = 100,
    ) -> None:
        """Configura os parâmetros principais do ACO."""
        self.n_ants = n_ants  # Número de formigas por iteração
        self.n_iterations = n_iterations  # Número máximo de iterações
        self.lower_bound = lower_bound  # Limite inferior do domínio
        self.upper_bound = upper_bound  # Limite superior do domínio
        self.evaporation_rate = evaporation_rate  # Taxa de evaporação do feromônio
        self.verbose = verbose  # Ativar logs detalhados
        self.early_stop_threshold = (
            early_stop_threshold  # Valor limite para parada antecipada
        )
        self.pheromone = np.ones(resolution)  # Mapa de feromônio (discreto)
        self.history: List[
            Tuple[int, Optional[float], float, np.ndarray, np.ndarray]
        ] = []  # Histórico de iterações

        if self.verbose:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
            )

    def run(self) -> Tuple[Optional[float], float]:
        """Executa o ciclo principal do ACO."""
        best_pos: Optional[float] = None
        best_fit: float = float("inf")

        for iteration in range(self.n_iterations):
            ants: List[Ant] = [
                Ant(self.lower_bound, self.upper_bound) for _ in range(self.n_ants)
            ]
            for ant in ants:
                ant.move(self.pheromone, self.lower_bound, self.upper_bound)

            positions = np.array([ant.position for ant in ants])
            fitness = np.array([ant.fitness for ant in ants])

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fit:
                best_fit = fitness[best_idx]
                best_pos = positions[best_idx]

            self.pheromone *= self.evaporation_rate
            for _, ant in enumerate(ants):
                idx = int(
                    (ant.position - self.lower_bound)
                    / (self.upper_bound - self.lower_bound)
                    * (len(self.pheromone) - 1)
                )
                idx = np.clip(idx, 0, len(self.pheromone) - 1)
                # Pondera reforço pelo fitness: quanto menor o f(x), maior o reforço
                self.pheromone[idx] += 1.0 / (1.0 + ant.fitness)

            self.history.append((iteration + 1, best_pos, best_fit, positions, fitness))

            if self.verbose:
                logging.info(
                    f"Iteration {iteration+1}: Best x = {best_pos:.5f}, f(x) = {best_fit:.8f}"
                )

            if best_fit <= self.early_stop_threshold:
                if self.verbose:
                    logging.info(
                        f"Parada antecipada na iteração {iteration+1} com f(x) = {best_fit:.8f}"
                    )
                break

        return best_pos, best_fit


def plot_history(
    history: List[Tuple[int, Optional[float], float, np.ndarray, np.ndarray]],
    upper_bound: float,
    lower_bound: float,
    save_path: Optional[str] = None,
) -> None:
    """Plota o histórico de todas as formigas e a trajetória da melhor solução."""
    x = np.linspace(lower_bound, upper_bound, 400)  # Intervalo para plotar f(x)
    y = f(x)
    plt.figure(figsize=(12, 7))
    plt.plot(x, y, label="f(x) = x^2", color="blue")

    # Desenha todas as posições das formigas (cinza)
    best_per_iter = []
    for _, _, _, positions, fitness in history:
        idx = np.argmin(fitness)
        best_per_iter.append((positions[idx], fitness[idx]))

    best_x_gray = [p[0] for p in best_per_iter]
    best_f_gray = [p[1] for p in best_per_iter]

    plt.scatter(
        best_x_gray, best_f_gray, color="gray", alpha=0.5, label="Melhor por iteração"
    )

    # Pega as melhores (vermelho)
    best_x = history[-1][1]
    best_f = history[-1][2]

    plt.scatter(
        cast(float, best_x),  # seguro que é float
        best_f,
        color="red",
        label="Melhores soluções",
    )
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.1, 0.5)
    plt.title("Histórico ACO: Todas as formigas")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
    else:
        plt.show()


def main() -> None:
    """Executa o ACO com argumentos da linha de comando e gera o gráfico."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations", type=int, default=100, help="Número de iterações"
    )
    parser.add_argument("--n_ants", type=int, default=50, help="Número de formigas")
    parser.add_argument(
        "--evaporation_rate", type=float, default=0.9, help="Taxa de evaporação"
    )
    parser.add_argument(
        "--early_stop_threshold",
        type=float,
        default=1e-5,
        help="Critério de parada antecipada",
    )
    parser.add_argument(
        "--upper_bound", type=float, default=10.0, help="Limite superior"
    )
    parser.add_argument(
        "--lower_bound", type=float, default=-10.0, help="Limite inferior"
    )
    parser.add_argument("--verbose", action="store_true", help="Ativar logs detalhados")
    parser.add_argument("--plot", action="store_true", help="Mostrar gráfico ao final")
    parser.add_argument(
        "--plot_upper_bound", type=float, default=10.0, help="Plotar limite superior"
    )
    parser.add_argument(
        "--plot_lower_bound", type=float, default=-10.0, help="Plotar limite inferior"
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="plots/aco_progress.png",
        help="Caminho para salvar o gráfico",
    )
    args = parser.parse_args()

    aco = ACO(
        n_ants=args.n_ants,
        n_iterations=args.iterations,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        evaporation_rate=args.evaporation_rate,
        verbose=args.verbose,
        early_stop_threshold=args.early_stop_threshold,
        resolution=args.n_ants * 10,
    )
    aco_x, aco_f = aco.run()

    print(f"ACO: Melhor x = {aco_x:.5f}, f(x) = {aco_f:.8f}")
    print("Real: x = 0.00000, f(x) = 0.00000")

    if args.plot or args.save_plot:
        plot_history(
            history=aco.history,
            upper_bound=args.upper_bound,
            lower_bound=args.lower_bound,
            save_path=args.save_plot if args.save_plot else None,
        )


if __name__ == "__main__":
    main()
