"""
Features:
- Feromônio 2D (tau[i,j]) na grade (w,n)
- Heurística (eta) configurável: "revenue" (receita) ou "yield" (produtividade)
- Restrições genéricas via penalidade (g(w,n) <= 0 é factível)
- Parada antecipada por estagnação
- Reprodutibilidade (seed)
- Artefatos: CSV do histórico, gráficos de convergência e heatmap
- Ótimo analítico contínuo (projetado) para receita (para checagem/validação)

---------------------------

# Meloeiro otimização receita
uv run aco_agro.py --crop melon --objective revenue --iters 1000 --ants 200 --rho 0.1

# Meloeiro otimização receita
uv run aco_agro.py --crop lettuce --objective revenue

# Meloeiro otimização produtividade
uv run aco_agro.py --crop melon --objective yield
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv
import json


@dataclass
class Crop:
    """
    Representa uma cultura com:
      - name: nome (apenas para logs/arquivos)
      - w_bounds: intervalo de lâmina de água (mm), ex.: (0.0, 700.0)
      - n_bounds: intervalo de dose de nitrogênio (kg/ha), ex.: (0.0, 350.0)
      - coeffs: coeficientes da função de produção quadrática:
          y(w,n) = a0 + a1*w + a2*n + a3*w^2 + a4*n^2 + a5*w*n
      - price_py: preço do produto (R$/kg)
      - cost_cw: custo da água (R$/ (mm·ha^-1))
      - cost_cn: custo do nitrogênio (R$/kg)
    """

    name: str
    w_bounds: Tuple[float, float]
    n_bounds: Tuple[float, float]
    coeffs: Tuple[float, float, float, float, float, float]
    price_py: float
    cost_cw: float
    cost_cn: float


def quadratic_yield(
    coeffs: Tuple[float, float, float, float, float, float],
    w: np.ndarray,
    n: np.ndarray,
) -> np.ndarray:
    """Calcula y(w,n) para uma função quadrática (ver definição em Crop)."""
    a0, a1, a2, a3, a4, a5 = coeffs
    return a0 + a1 * w + a2 * n + a3 * (w**2) + a4 * (n**2) + a5 * w * n


def revenue(crop: Crop, w: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Receita líquida R(w,n) = p_y * y(w,n) - c_w * w - c_n * n."""
    y = quadratic_yield(crop.coeffs, w, n)
    return crop.price_py * y - crop.cost_cw * w - crop.cost_cn * n


def make_grid(bounds: Tuple[float, float], steps: int) -> np.ndarray:
    """Cria uma grade 1D (linspace) que inclui limites inferior e superior."""
    return np.linspace(bounds[0], bounds[1], steps)


def analytic_optimum_for_revenue(crop: Crop) -> Tuple[float, float]:
    """
    Calcula o ótimo contínuo (sem restrição) para a receita e projeta no domínio caixa.

    dR/dw = p*(a1 + 2*a3*w + a5*n) - cw = 0
    dR/dn = p*(a2 + 2*a4*n + a5*w) - cn = 0

    Se o sistema for singular, usa o centro do domínio. Em seguida, projeta (w*,n*)
    para dentro de [w_min,w_max] × [n_min,n_max] (borda ativa, se necessário).
    """
    a0, a1, a2, a3, a4, a5 = crop.coeffs
    p, cw, cn = crop.price_py, crop.cost_cw, crop.cost_cn

    A = np.array([[2 * p * a3, p * a5], [p * a5, 2 * p * a4]], dtype=float)
    b = np.array([cw - p * a1, cn - p * a2], dtype=float)

    try:
        sol = np.linalg.solve(A, b)
        w_star, n_star = sol[0], sol[1]
    except np.linalg.LinAlgError:
        # Sistema degenerado: cai para o centro do domínio
        w_star = (crop.w_bounds[0] + crop.w_bounds[1]) / 2
        n_star = (crop.n_bounds[0] + crop.n_bounds[1]) / 2

    # Projeta no domínio caixa (respeita limites)
    w_proj = np.clip(w_star, crop.w_bounds[0], crop.w_bounds[1])
    n_proj = np.clip(n_star, crop.n_bounds[0], crop.n_bounds[1])
    return float(w_proj), float(n_proj)


class Constraint:
    """
    Restrição genérica do tipo g(w,n) <= 0 (factível).
    Se g(w,n) > 0, aplica-se penalidade = weight * g(w,n).

    Exemplo: teto de orçamento -> g(w,n) = (c_w*w + c_n*n) - B
    """

    def __init__(self, func: Callable[[float, float], float], weight: float, name: str):
        self.func = func
        self.weight = weight
        self.name = name

    def penalty(self, w: float, n: float) -> float:
        """Retorna penalidade >= 0 para (w,n)."""
        val = self.func(w, n)
        return max(0.0, val) * self.weight


class ACO2D:
    """
    ACO com feromônio 2D sobre a grade (w,n).

    Parâmetros
    ----------
    crop : Crop
        Definição do domínio, função de produção e parâmetros econômicos.
    objective : {"revenue","yield"}
        Objetivo a otimizar (receita ou produtividade).
    steps_w, steps_n : int
        Resolução da grade (pontos em cada eixo).
    ants : int
        Nº de amostras por iteração.
    iterations : int
        Nº máximo de iterações.
    alpha, beta : float
        Pesos para feromônio (tau) e heurística (eta).
    rho : float
        Taxa de evaporação: tau <- (1-rho)*tau + depósito.
    q : float
        Escala do depósito.
    elitist_weight : float
        Reforço adicional na melhor global por iteração.
    topk_fraction : float
        Fração de melhores da iteração que depositam.
    seed : int | None
        Semente aleatória (reprodutibilidade).
    constraints : list[Constraint] | None
        Restrições do tipo g(w,n) <= 0 via penalidade.
    penalty_scale : float
        Escala global aplicada às penalidades.
    early_stop_tol : float
        Tolerância (não usada diretamente no delta, mas com “patience”).
    early_stop_patience : int
        Nº de iterações sem melhora até interromper.
    out_dir : str
        Pasta para CSV/figuras/JSON.
    """

    def __init__(
        self,
        crop: Crop,
        objective: str = "revenue",
        steps_w: int = 701,
        steps_n: int = 351,
        ants: int = 200,
        iterations: int = 1000,
        alpha: float = 1.0,
        beta: float = 1.0,
        rho: float = 0.1,
        q: float = 1.0,
        elitist_weight: float = 2.0,
        topk_fraction: float = 0.2,
        seed: Optional[int] = None,
        constraints: Optional[List[Constraint]] = None,
        penalty_scale: float = 1.0,
        early_stop_tol: float = 1e-6,
        early_stop_patience: int = 100,
        out_dir: str = "outputs",
    ):
        assert objective in ("revenue", "yield")
        self.crop = crop
        self.objective = objective
        self.steps_w = steps_w
        self.steps_n = steps_n
        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.elitist_weight = elitist_weight
        self.topk_fraction = topk_fraction
        self.rng = np.random.default_rng(seed)
        self.constraints = constraints or []
        self.penalty_scale = penalty_scale
        self.early_stop_tol = early_stop_tol
        self.early_stop_patience = early_stop_patience
        self.out_dir = out_dir

        # --- GRADE ---
        self.W_vals = make_grid(crop.w_bounds, steps_w)
        self.N_vals = make_grid(crop.n_bounds, steps_n)
        self.W_grid, self.N_grid = np.meshgrid(self.W_vals, self.N_vals)

        # --- HEURÍSTICA eta (normalizada para (0,1]) ---
        if objective == "revenue":
            eta_raw = revenue(crop, self.W_grid, self.N_grid)
        else:
            eta_raw = quadratic_yield(crop.coeffs, self.W_grid, self.N_grid)

        # Aplica penalidades diretamente em eta_raw (evita regiões inviáveis)
        if self.constraints:
            pen = np.zeros_like(eta_raw, dtype=float)
            for c in self.constraints:
                pen += c.weight * np.maximum(0.0, c.func(self.W_grid, self.N_grid))
            eta_raw = eta_raw - self.penalty_scale * pen

        # Normalização para (0,1] para uso em probabilidade
        min_eta = float(np.min(eta_raw))
        max_eta = float(np.max(eta_raw))
        eps = 1e-9
        if max_eta - min_eta < 1e-12:
            eta_norm = np.ones_like(eta_raw)
        else:
            eta_norm = (eta_raw - min_eta) / (max_eta - min_eta) * (1 - eps) + eps

        self.eta = eta_norm

        # --- FEROMÔNIO ---
        self.tau = np.ones_like(self.eta, dtype=float)

        # --- HISTÓRICO ---
        self.history: List[Dict] = []
        self.global_best = {"w": None, "n": None, "score": -np.inf, "i": 0}

    def _score_point(self, w: float, n: float) -> float:
        """
        Retorna o valor do objetivo (receita ou produtividade) menos penalidades (se houver).
        """
        if self.objective == "revenue":
            base = float(revenue(self.crop, np.array([w]), np.array([n]))[0])
        else:
            base = float(
                quadratic_yield(self.crop.coeffs, np.array([w]), np.array([n]))[0]
            )

        # Penalidades por violação de restrições (g>0)
        total_penalty = 0.0
        for c in self.constraints:
            total_penalty += max(0.0, c.func(w, n)) * c.weight

        return base - self.penalty_scale * total_penalty

    def run(self, verbose: bool = True) -> Dict:
        """
        Executa o loop principal do ACO e retorna a melhor solução global.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        best_score = -np.inf
        no_improve = 0

        flat_size = self.tau.size
        topk = max(1, int(self.topk_fraction * self.ants))

        for it in range(1, self.iterations + 1):
            # --- Probabilidades de amostragem: (tau^alpha * eta^beta) ---
            weights = np.power(self.tau, self.alpha) * np.power(self.eta, self.beta)
            weights = weights.ravel()
            wsum = float(weights.sum())

            if wsum <= 0 or not np.isfinite(wsum):
                # Caso degenerado: substitui por distribuição uniforme
                weights = np.ones_like(weights) / flat_size
            else:
                weights = weights / wsum

            # --- Amostragem de formigas (com reposição) no plano 2D ---
            idxs = self.rng.choice(flat_size, size=self.ants, replace=True, p=weights)
            i_idx = idxs // self.steps_w
            j_idx = idxs % self.steps_w

            ws = self.W_vals[j_idx]
            ns = self.N_vals[i_idx]
            scores = np.array(
                [self._score_point(float(w), float(n)) for w, n in zip(ws, ns)],
                dtype=float,
            )

            # Melhor da iteração
            k_best = int(np.argmax(scores))
            iter_best_w = float(ws[k_best])
            iter_best_n = float(ns[k_best])
            iter_best_score = float(scores[k_best])

            # Atualiza melhor global
            if iter_best_score > best_score + 1e-15:
                best_score = iter_best_score
                no_improve = 0
                self.global_best = {
                    "w": iter_best_w,
                    "n": iter_best_n,
                    "score": iter_best_score,
                    "i": it,
                }
            else:
                no_improve += 1

            # --- Evaporação global ---
            self.tau *= 1.0 - self.rho

            # --- Depósito top-k normalizado por desempenho ---
            topk_idx = np.argpartition(scores, -topk)[-topk:]
            s_min = float(np.min(scores))
            s_max = float(np.max(scores))
            denom = (s_max - s_min) if (s_max - s_min) > 1e-12 else 1.0

            for k in topk_idx:
                ii = int(i_idx[k])
                jj = int(j_idx[k])
                contrib = self.q * (scores[k] - s_min) / denom
                self.tau[ii, jj] += contrib

            # --- Reforço elitista no melhor global ---
            gb_w = self.global_best["w"]
            gb_n = self.global_best["n"]
            if gb_w is not None and gb_n is not None:
                jj = int(np.argmin(np.abs(self.W_vals - gb_w)))
                ii = int(np.argmin(np.abs(self.N_vals - gb_n)))
                self.tau[ii, jj] += self.elitist_weight * self.q

            # Evita problemas numéricos extremos
            self.tau = np.clip(self.tau, 1e-6, 1e6)

            # --- Log do histórico ---
            self.history.append(
                {
                    "iteration": it,
                    "iter_best_w": iter_best_w,
                    "iter_best_n": iter_best_n,
                    "iter_best_score": iter_best_score,
                    "global_best_w": self.global_best["w"],
                    "global_best_n": self.global_best["n"],
                    "global_best_score": self.global_best["score"],
                }
            )

            if no_improve >= self.early_stop_patience and self.early_stop_tol > 0:
                if verbose:
                    print(
                        f"[EarlyStop] No improvement for {self.early_stop_patience} iterations at it={it}."
                    )
                break

            if verbose and (it % max(1, self.iterations // 10) == 0 or it <= 3):
                print(
                    f"[{self.crop.name}] it={it:4d} | iter_best=({iter_best_w:.2f},{iter_best_n:.2f}) "
                    f"s={iter_best_score:.4f} | global=({self.global_best['w']:.2f},{self.global_best['n']:.2f}) "
                    f"S*={self.global_best['score']:.4f}"
                )

        return self.global_best

    def save_history_csv(self, path: str) -> None:
        """Salva o histórico (uma linha por iteração) em CSV."""
        if not self.history:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

    def plot_convergence(self, path: Optional[str] = None) -> None:
        """Plota as curvas do melhor da iteração e do melhor global (convergência)."""
        if not self.history:
            return
        iters = [h["iteration"] for h in self.history]
        best_iter_scores = [h["iter_best_score"] for h in self.history]
        global_best_scores = [h["global_best_score"] for h in self.history]

        subObjective = (
            "Receita líquida" if self.objective == "revenue" else self.objective
        )
        subObjective = "Produtividade" if self.objective == "yield" else self.objective

        plt.figure(figsize=(9, 6))
        plt.plot(iters, best_iter_scores, label="Best of iteration")
        plt.plot(iters, global_best_scores, label="Global best", linestyle="--")
        plt.xlabel("Iteration")
        plt.ylabel("Objective value")
        plt.title(f"ACO Convergence — {self.crop.name} ({subObjective})")
        plt.legend()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_heatmap(
        self,
        path: Optional[str] = None,
        mark_history: bool = True,
        cmap: str = "cividis",
        marker_size: int = 60,
        marker_alpha: float = 0.9,
    ) -> None:
        """
        Plota um **mapa de calor (heatmap)** do objetivo (receita ou produtividade),
        destacando claramente os **melhores pontos por iteração** e o **ótimo global**.

        - Utiliza colormap "cividis" (bom contraste e acessível a daltônicos).
        - Marcadores grandes e opacos para melhor visibilidade.
        - Sem linhas de trajetória (somente pontos).
        """

        if self.objective == "revenue":
            Z = revenue(self.crop, self.W_grid, self.N_grid)
        else:
            Z = quadratic_yield(self.crop.coeffs, self.W_grid, self.N_grid)

        plt.figure(figsize=(10, 8))
        cs = plt.contourf(self.W_vals, self.N_vals, Z, levels=60, cmap=cmap)
        cbar = plt.colorbar(cs)
        cbar.set_label("Valor do objetivo", rotation=90)

        if mark_history and self.history:
            ws = [h["iter_best_w"] for h in self.history]
            ns = [h["iter_best_n"] for h in self.history]
            plt.scatter(
                ws,
                ns,
                s=marker_size,
                alpha=marker_alpha,
                color="gray",
                edgecolor="black",
                linewidths=0.8,
                label="Melhores por iteração",
                zorder=5,
            )

        gb_w = self.global_best["w"]
        gb_n = self.global_best["n"]
        if gb_w is not None and gb_n is not None:
            plt.scatter(
                [gb_w],
                [gb_n],
                marker="*",
                s=300,
                color="red",
                edgecolor="black",
                linewidths=1.2,
                label="Melhor global",
                zorder=6,
            )

        subObjective = (
            "Receita líquida" if self.objective == "revenue" else self.objective
        )
        subObjective = "Produtividade" if self.objective == "yield" else self.objective

        plt.xlabel("Lâmina de água w (mm)")
        plt.ylabel("Dose de nitrogênio n (kg/ha)")
        plt.title(f"Mapa de calor do objetivo — {self.crop.name} ({subObjective})")
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()

        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=160, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_surface_3d(
        self,
        path: Optional[str] = None,
        mark_history: bool = True,
        elev: float = 30.0,
        azim: float = 135.0,
        cmap: str = "viridis",
        alpha: float = 0.85,
        marker_size: int = 70,
        point_color: str = "gray",
        point_edge: str = "black",
        lift_frac: float = 0.01,
        use_halo: bool = True,
        halo_factor: float = 1.8,
        halo_color: str = "white",
    ) -> None:
        """
        Superfície 3D do objetivo com pontos **sempre visíveis**.

        Truques usados para evitar que os pontos "afundem" na superfície:
        - `depthshade=False` nos pontos (desabilita sombreamento que "lava" a cor).
        - `shade=False` na superfície (menos sombreado competindo com os pontos).
        - `zorder` maior nos pontos e menor na superfície.
        - **"Lift"**: eleva os pontos um pouquinho acima de Z (fração do range).
        - **Halo**: desenha um ponto branco maior por trás (efeito contorno/brilho).
        - Borda preta nos marcadores.
        """

        if self.objective == "revenue":
            Z = revenue(self.crop, self.W_grid, self.N_grid)
        else:
            Z = quadratic_yield(self.crop.coeffs, self.W_grid, self.N_grid)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            self.W_grid,
            self.N_grid,
            Z,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            edgecolor="none",
            alpha=alpha,
            shade=False,
            zorder=0,
        )

        if mark_history and self.history:
            ws = np.array([h["iter_best_w"] for h in self.history], dtype=float)
            ns = np.array([h["iter_best_n"] for h in self.history], dtype=float)
            if self.objective == "revenue":
                zs = revenue(self.crop, ws, ns)
            else:
                zs = quadratic_yield(self.crop.coeffs, ws, ns)

            z_range = float(np.nanmax(Z) - np.nanmin(Z)) or 1.0
            zs_lifted = zs + lift_frac * z_range

            if use_halo:
                ax.scatter(
                    ws,
                    ns,
                    zs_lifted,
                    s=marker_size * halo_factor,
                    alpha=0.95,
                    c=halo_color,
                    edgecolors="none",
                    depthshade=False,
                    zorder=8,  # desenha antes do ponto real (fica "atrás")
                )

            # ░░░ Ponto real (com borda preta) ░░░
            ax.scatter(
                ws,
                ns,
                zs_lifted,
                s=marker_size,
                alpha=0.95,
                c=point_color,
                edgecolors=point_edge,
                linewidths=0.9,
                depthshade=False,  # impede sombreamento que reduz contraste
                zorder=9,  # acima da superfície e do halo
                label="Melhores por iteração",
            )

        # --- Melhor global (estrela) com mesmo "lift" ---
        gb_w = self.global_best["w"]
        gb_n = self.global_best["n"]
        if gb_w is not None and gb_n is not None:
            if self.objective == "revenue":
                gb_z = float(revenue(self.crop, np.array([gb_w]), np.array([gb_n]))[0])
            else:
                gb_z = float(
                    quadratic_yield(
                        self.crop.coeffs, np.array([gb_w]), np.array([gb_n])
                    )[0]
                )
            gb_z_lift = gb_z + lift_frac * (float(np.nanmax(Z) - np.nanmin(Z)) or 1.0)

            # Halo da estrela (opcional) — ajuda a destacar no 3D
            if use_halo:
                ax.scatter(
                    [gb_w],
                    [gb_n],
                    [gb_z_lift],
                    marker="o",
                    s=350 * halo_factor,
                    c=halo_color,
                    edgecolors="none",
                    alpha=0.95,
                    depthshade=False,
                    zorder=10,
                )

            ax.scatter(
                [gb_w],
                [gb_n],
                [gb_z_lift],
                marker="*",
                s=420,
                c="red",
                edgecolors="black",
                linewidths=1.2,
                depthshade=False,
                zorder=11,
                label="Melhor global",
            )

        # --- Eixos, título, colorbar ---
        subObjective = (
            "Receita líquida" if self.objective == "revenue" else "Produtividade"
        )
        ax.set_xlabel("Lâmina de água w (mm)")
        ax.set_ylabel("Dose de nitrogênio n (kg/ha)")
        ax.set_zlabel("Valor do objetivo")
        ax.set_title(f"Superfície 3D — {self.crop.name} ({subObjective})")
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper left")

        fig.colorbar(surf, shrink=0.6, aspect=14, pad=0.08)
        plt.tight_layout()

        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=170, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)


# Lettuce (Alface-americana)
# y_alf = -12.490 + 388.1*w - 6.02*n - 1.042*w^2 - 0.04563*n^2 + 0.1564*w*n
lettuce = Crop(
    name="Alface-americana",
    w_bounds=(0.0, 250.0),
    n_bounds=(100.0, 240.0),
    coeffs=(-12.490, 388.1, -6.02, -1.042, -0.04563, 0.1564),
    price_py=0.80,
    cost_cw=0.44,
    cost_cn=2.09,
)

# Melon (Meloeiro)
# y_mel = 34.16737*n + 70.77509*w - 0.05781*w^2 - 0.07612*n^2
melon = Crop(
    name="Meloeiro",
    w_bounds=(0.0, 700.0),
    n_bounds=(0.0, 350.0),
    coeffs=(0.0, 70.77509, 34.16737, -0.05781, -0.07612, 0.0),
    price_py=0.40,
    cost_cw=0.134,
    cost_cn=2.33,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ACO 2D for crops (revenue or yield) with quadratic production function and optional constraints."
    )
    parser.add_argument(
        "--crop",
        choices=["lettuce", "melon"],
        default="melon",
        help="Crop to optimize.",
    )
    parser.add_argument(
        "--objective",
        choices=["revenue", "yield"],
        default="revenue",
        help="Objective function.",
    )
    parser.add_argument(
        "--steps_w", type=int, default=701, help="Grid points for water depth (w)."
    )
    parser.add_argument(
        "--steps_n", type=int, default=351, help="Grid points for nitrogen dose (n)."
    )
    parser.add_argument(
        "--ants", type=int, default=200, help="Number of ants (samples) per iteration."
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="Maximum number of iterations."
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Pheromone weight (tau^alpha)."
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Heuristic weight (eta^beta)."
    )
    parser.add_argument(
        "--rho", type=float, default=0.1, help="Evaporation rate 0<rho<1."
    )
    parser.add_argument("--q", type=float, default=1.0, help="Pheromone deposit scale.")
    parser.add_argument(
        "--elitist_weight",
        type=float,
        default=2.0,
        help="Elitist reinforcement weight per iteration.",
    )
    parser.add_argument(
        "--topk_fraction",
        type=float,
        default=0.2,
        help="Fraction of top ants depositing pheromone.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (reproducibility)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Output directory for CSV/plots/json.",
    )
    args = parser.parse_args()

    crop = lettuce if args.crop == "lettuce" else melon

    if args.crop == "lettuce" and args.steps_w == 701 and args.steps_n == 351:
        args.steps_w = 251
        args.steps_n = 141

    aco = ACO2D(
        crop=crop,
        objective=args.objective,
        steps_w=args.steps_w,
        steps_n=args.steps_n,
        ants=args.ants,
        iterations=args.iters,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        q=args.q,
        elitist_weight=args.elitist_weight,
        topk_fraction=args.topk_fraction,
        seed=args.seed,
        constraints=[],
        penalty_scale=1.0,
        early_stop_tol=1e-6,
        early_stop_patience=100,
        out_dir=args.out_dir,
    )

    best = aco.run(verbose=True)

    base = os.path.join(
        args.out_dir, f"{crop.name.replace(' ', '_').lower()}_{args.objective}"
    )
    os.makedirs(args.out_dir, exist_ok=True)
    aco.save_history_csv(base + "_history.csv")
    aco.plot_convergence(base + "_convergence.png")
    aco.plot_heatmap(base + "_heatmap.png")
    aco.plot_surface_3d(base + "_surface3d.png")

    meta = {
        "crop": crop.name,
        "objective": args.objective,
        "best": best,
        "analytic_optimum_projected_for_revenue": (
            analytic_optimum_for_revenue(crop) if args.objective == "revenue" else None
        ),
    }
    with open(base + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Best solution:", best)
