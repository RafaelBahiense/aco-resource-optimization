import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore [import-untyped]
import typing
from typing import List, Tuple, Optional, overload


@overload
def melon_yield(w: float, n: float) -> float: ...
@overload
def melon_yield(w: np.ndarray, n: np.ndarray) -> np.ndarray: ...
def melon_yield(w, n):
    """
    Calcula a função de produção do meloeiro.

    Parâmetros:
    - w: Lâmina de água (mm)
    - n: Dose de nitrogênio (kg/ha)

    Retorna:
    - Produtividade estimada para os valores fornecidos.
    """
    return 34.16737 * n + 70.77509 * w - 0.05781 * w**2 - 0.07612 * n**2


class Ant:
    """
    Representa uma formiga no algoritmo ACO para o meloeiro.
    """

    def __init__(
        self,
        w_slots: np.ndarray,
        n_slots: np.ndarray,
        w_bounds: Tuple[float, float],
        n_bounds: Tuple[float, float],
    ) -> None:
        """
        Inicializa a formiga escolhendo slots e calculando produtividade.
        """
        self.w_idx: int = np.random.choice(len(w_slots), p=w_slots / w_slots.sum())
        self.n_idx: int = np.random.choice(len(n_slots), p=n_slots / n_slots.sum())

        self.w: float = w_bounds[0] + self.w_idx * (w_bounds[1] - w_bounds[0]) / (
            len(w_slots) - 1
        )
        self.n: float = n_bounds[0] + self.n_idx * (n_bounds[1] - n_bounds[0]) / (
            len(n_slots) - 1
        )

        self.yield_value: float = melon_yield(self.w, self.n)


class ACO:
    """
    Algoritmo Ant Colony Optimization para otimizar a função de produção do meloeiro.
    """

    def __init__(
        self,
        n_ants: int,
        n_iterations: int,
        w_bounds: Tuple[float, float] = (0, 700),
        n_bounds: Tuple[float, float] = (0, 350),
        evaporation_rate: float = 0.9,
        early_stop_threshold: float = 1e-5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Inicializa o ACO com os parâmetros principais.
        """
        self.w_length: int = int(w_bounds[1] - w_bounds[0])
        self.n_length: int = int(n_bounds[1] - n_bounds[0])

        self.w_slots: np.ndarray = np.ones(self.w_length)
        self.n_slots: np.ndarray = np.ones(self.n_length)

        self.w_bounds: Tuple[float, float] = w_bounds
        self.n_bounds: Tuple[float, float] = n_bounds

        self.evaporation: float = evaporation_rate
        self.early_stop_threshold: float = early_stop_threshold

        self.n_ants: int = n_ants
        self.n_iterations: int = n_iterations

        self.history: List[Tuple[int, float, float, float]] = []
        self.best_ant: Optional[Ant] = None

        self.logger = logger

    def run(self) -> None:
        """
        Executa o loop principal de iterações do ACO.
        """
        for iteration in range(self.n_iterations):
            ants: List[Ant] = [
                Ant(self.w_slots, self.n_slots, self.w_bounds, self.n_bounds)
                for _ in range(self.n_ants)
            ]

            best_ant_iteration: Ant = max(ants, key=lambda a: a.yield_value)

            if (
                self.best_ant is None
                or best_ant_iteration.yield_value > self.best_ant.yield_value
            ):
                self.best_ant = best_ant_iteration

            self.w_slots *= self.evaporation
            self.n_slots *= self.evaporation

            for ant in ants:
                self.w_slots[ant.w_idx] += ant.yield_value / self.best_ant.yield_value
                self.n_slots[ant.n_idx] += ant.yield_value / self.best_ant.yield_value

            self.history.append(
                (
                    iteration + 1,
                    best_ant_iteration.w,
                    best_ant_iteration.n,
                    best_ant_iteration.yield_value,
                )
            )

            if self.logger is not None:
                self.logger.info(
                    f"Iteração {iteration+1}: Melhor w={best_ant_iteration.w:.2f} mm, "
                    f"n={best_ant_iteration.n:.2f} kg/ha, "
                    f"Produtividade={best_ant_iteration.yield_value:.2f}"
                )

        if self.best_ant is None:
            raise ValueError("Nenhuma formiga encontrou uma solução viável.")

        print(
            f"\n\nMelhor solução encontrada:\n"
            f"Lâmina de água (w): {self.best_ant.w:.2f} mm\n"
            f"Dose de nitrogênio (n): {self.best_ant.n:.2f} kg/ha\n"
            f"Produtividade estimada: {self.best_ant.yield_value:.2f}\n"
        )

    def plot_heatmap(
        self,
        save_path: Optional[str] = None,
        save_3d_path: Optional[str] = "plots/aco_meloeiro_3D.png",
    ) -> None:
        """
        Gera:
        - Heatmap 2D com melhores de cada iteração (cinza) e melhor final (vermelho)
        - Superfície 3D com a trajetória
        """

        # HEATMAP 2D
        W = np.linspace(*self.w_bounds, 100)
        N = np.linspace(*self.n_bounds, 100)
        W_grid, N_grid = np.meshgrid(W, N)
        Z = melon_yield(W_grid, N_grid)

        plt.figure(figsize=(10, 7))
        cp = plt.contourf(W, N, Z, levels=50, cmap="viridis")
        plt.colorbar(cp)

        for _, w, n, _ in self.history:
            plt.plot(w, n, "o", color="gray", alpha=0.5)

        assert self.best_ant is not None
        plt.plot(self.best_ant.w, self.best_ant.n, "r*", markersize=15)

        plt.xlabel("Lâmina de Água (mm)")
        plt.ylabel("Dose de Nitrogênio (kg/ha)")
        plt.title("Função de Produção do Meloeiro + Melhor Solução ACO")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Heatmap 2D salvo em: {save_path}")
        else:
            plt.show()
        plt.close()

        # SUPERFÍCIE 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax = typing.cast(Axes3D, ax)
        surf = ax.plot_surface(W_grid, N_grid, Z, cmap="viridis", alpha=0.8)

        ax.scatter(
            self.best_ant.w,
            self.best_ant.n,
            int(melon_yield(self.best_ant.w, self.best_ant.n)),
            color="red",
            s=80,
            label="Melhor Solução Final",
        )

        history_w = [w for _, w, _, _ in self.history]
        history_n = [n for _, _, n, _ in self.history]
        history_y = [melon_yield(w, n) for w, n in zip(history_w, history_n)]

        ax.scatter(
            history_w,
            history_n,
            history_y,
            color="gray",
            alpha=0.4,
            label="Melhores por Iteração",
        )

        ax.set_xlabel("Lâmina de Água (mm)")
        ax.set_ylabel("Dose de Nitrogênio (kg/ha)")
        ax.set_zlabel("Produtividade")
        ax.set_title("Função de Produção do Meloeiro (Superfície 3D)")

        fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.legend()

        ax.view_init(elev=30, azim=145)

        if save_3d_path:
            os.makedirs(os.path.dirname(save_3d_path), exist_ok=True)
            fig.savefig(save_3d_path)
            print(f"Superfície 3D salva em: {save_3d_path}")

        # plt.show()
        plt.close(fig)


def main() -> None:
    """
    Executa o ACO via linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Otimização do manejo do meloeiro com ACO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Grupo: Execução do algoritmo
    group_algo = parser.add_argument_group("Parâmetros do Algoritmo")
    group_algo.add_argument(
        "--iterations", type=int, default=200, help="Número de iterações"
    )
    group_algo.add_argument("--n_ants", type=int, default=50, help="Número de formigas")
    group_algo.add_argument(
        "--evaporation_rate", type=float, default=0.9, help="Taxa de evaporação"
    )

    # Grupo: Logs e visualização
    group_output = parser.add_argument_group("Saída e Visualização")
    group_output.add_argument(
        "--verbose", action="store_true", help="Ativar logs detalhados"
    )
    group_output.add_argument(
        "--plot", action="store_true", help="Mostrar gráfico ao final"
    )
    group_output.add_argument(
        "--save_plot",
        type=str,
        default="plots/aco_meloeiro.png",
        metavar="CAMINHO",
        help="Salvar heatmap",
    )

    args = parser.parse_args()

    logLevel = logging.INFO if args.verbose else logging.ERROR

    logging.basicConfig(
        level=logLevel, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    aco = ACO(
        n_ants=args.n_ants,
        n_iterations=args.iterations,
        evaporation_rate=args.evaporation_rate,
        logger=logging.getLogger("ACO_Meloeiro"),
    )
    aco.run()

    if args.plot or args.save_plot:
        aco.plot_heatmap(save_path=args.save_plot if args.save_plot else None)


if __name__ == "__main__":
    main()
