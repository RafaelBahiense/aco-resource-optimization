"""
ACO contínuo (ACOR) para estimação de harmônicos em Sistemas Elétricos de Potência.

Replica o cenário de RABÊLO et al. (2011) — "Uma Aplicação de PSO na Qualidade da
Energia Elétrica" — que estima os componentes harmônicos de sinais de tensão/corrente
modelados por uma série de Fourier com componente contínua (CC) de decaimento exponencial.

Modelo do sinal (fases nulas, como nos casos de teste do paper):

    x(t) = x0 * exp(-lambda * t) + sum_{i=1..H} [ Ac_i * cos(i*w0*t) + As_i * sin(i*w0*t) ]

A estimação é tratada como otimização contínua em R^(2+2H): cada "formiga" propõe um
vetor de parâmetros (x0, lambda, Ac_1, As_1, ..., Ac_H, As_H) e a qualidade é o inverso
do erro de reconstrução (RMSE) entre o sinal sintético de referência e o reconstruído
(Eq. 6 do paper).

Diferente do `aco_agro.py` (feromônio em grade 2D), aqui a dimensionalidade (16 para
H=7) inviabiliza a grade, então usa-se ACOR (SOCHA; DORIGO, 2008): um arquivo de
soluções + kernels gaussianos por dimensão substitui a matriz de feromônio.

Convenção de tempo: NORMALIZADA — a janela de 1 ciclo corresponde a t em [0,1), com a
fundamental igual a cos(2*pi*t). Assim, lambda=0.4 produz ~33% de decaimento na janela,
tornando a constante de decaimento identificável (consistente com os resultados do paper).

---------------------------

# Estimação na forma de onda da tensão (caso do paper)
uv run aco_eletrica.py --signal voltage

# Estimação na corrente, com mais iterações
uv run aco_eletrica.py --signal current --iters 3000

# Cenário com ruído (robustez além do baseline do paper)
uv run aco_eletrica.py --signal voltage --noise_snr 30
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json


TIME_CONVENTION = (
    "Tempo normalizado: uma janela de 1 ciclo corresponde a t em [0, 1), "
    "com fundamental cos(2*pi*t)."
)
PARAMETER_VECTOR_ORDER = "[x0, lambda, Ac_1, As_1, ..., Ac_H, As_H]"
DATA_SOURCE = (
    "Cenários sintéticos de tensão/corrente de Rabêlo et al. (2011), "
    "Eqs. 7 e 8, com 7 harmônicos e componente CC com decaimento exponencial."
)
PUBLISHED_BASELINE = {
    "voltage": {
        "pso_mean_err_pct": 0.0290,
        "tdf_mean_err_pct": 7.0475,
    },
    "current": {
        "pso_mean_err_pct": 0.0290,
        "tdf_mean_err_pct": 7.1101,
    },
}


@dataclass
class HarmonicSignal:
    """
    Instância do problema: um sinal periódico distorcido (ground truth).

      - name: rótulo (apenas para logs/arquivos).
      - x0: amplitude da componente contínua (CC).
      - lam: constante de decaimento (lambda) da CC, em tempo normalizado.
      - harmonics: lista de pares (Ac_i, As_i) — amplitudes cosseno/seno por harmônico,
        do 1º (fundamental) ao H-ésimo.

    O número de harmônicos H é len(harmonics). O sinal é um benchmark sintético de
    coeficientes conhecidos; ruído de medição é adicionado opcionalmente em
    `generate_signal`.
    """

    name: str
    x0: float
    lam: float
    harmonics: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def H(self) -> int:
        """Número de harmônicos do sinal."""
        return len(self.harmonics)

    def true_vector(self) -> np.ndarray:
        """Vetor de parâmetros verdadeiro [x0, lam, Ac_1, As_1, ..., Ac_H, As_H]."""
        vec = [self.x0, self.lam]
        for ac, as_ in self.harmonics:
            vec.extend([ac, as_])
        return np.array(vec, dtype=float)


def sample_times(samples: int, cycles: float = 1.0) -> np.ndarray:
    """
    Vetor de tempos normalizado: `samples` pontos cobrindo `cycles` ciclos da fundamental.

    Tempo normalizado => 1 ciclo equivale a t indo de 0 a 1 (fundamental = cos(2*pi*t)).
    Os pontos são igualmente espaçados, sem repetir o início do próximo ciclo.
    """
    return np.linspace(0.0, cycles, samples, endpoint=False)


def harmonic_basis(t: np.ndarray, H: int) -> np.ndarray:
    """
    Matriz de base seno/cosseno B de forma (len(t), 2H), fixa (independe dos parâmetros).

    Colunas, por harmônico i=1..H: [cos(2*pi*i*t), sin(2*pi*i*t)].
    A reconstrução harmônica de um conjunto de amplitudes A (..., 2H) é A @ B.T.
    """
    w0 = 2.0 * np.pi
    cols = []
    for i in range(1, H + 1):
        cols.append(np.cos(i * w0 * t))
        cols.append(np.sin(i * w0 * t))
    return np.stack(cols, axis=1)


def reconstruct(theta: np.ndarray, t: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Reconstrói x̂(t) a partir de vetores de parâmetros (vetorizado).

    Aceita `theta` 1D (um candidato, forma (2+2H,)) ou 2D (vários candidatos, forma
    (P, 2+2H)). Retorna forma (len(t),) ou (P, len(t)), respectivamente.

    A parte harmônica (A @ basis.T) usa a base pré-computada; só a CC com decaimento
    (x0 * exp(-lam * t)) é recalculada, pois depende de lambda.
    """
    single = theta.ndim == 1
    th = np.atleast_2d(theta)  # (P, 2+2H)
    x0 = th[:, 0:1]  # (P,1)
    lam = th[:, 1:2]  # (P,1)
    amps = th[:, 2:]  # (P,2H)

    dc = x0 * np.exp(-lam * t[None, :])  # (P, m)
    harm = amps @ basis.T  # (P, m)
    recon = dc + harm  # (P, m)
    return recon[0] if single else recon


def rmse(theta: np.ndarray, t: np.ndarray, basis: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Raiz do erro quadrático médio entre sinal reconstruído e medido (por candidato)."""
    recon = reconstruct(theta, t, basis)
    diff = recon - measured  # broadcasting com measured (m,)
    if diff.ndim == 1:
        return float(np.sqrt(np.mean(diff**2)))
    return np.sqrt(np.mean(diff**2, axis=1))


def fitness(rmse_val: np.ndarray, delta: float = 1e-5) -> np.ndarray:
    """Função de aptidão do paper (Eq. 6): FA = 1 / (RMSE + delta). Maior é melhor."""
    return 1.0 / (rmse_val + delta)


def generate_signal(
    signal: HarmonicSignal,
    t: np.ndarray,
    noise_snr_db: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Gera o sinal sintético observado a partir da instância (ground truth), com ruído
    gaussiano opcional.

    Se `noise_snr_db` for informado, adiciona ruído branco gaussiano com a SNR pedida
    (em dB), calculada sobre a potência do sinal limpo.
    """
    H = signal.H
    basis = harmonic_basis(t, H)
    clean = reconstruct(signal.true_vector(), t, basis)

    if noise_snr_db is None:
        return clean

    if rng is None:
        rng = np.random.default_rng()
    sig_power = float(np.mean(clean**2))
    noise_power = sig_power / (10.0 ** (noise_snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=clean.shape)
    return clean + noise


def tdf_estimate(measured: np.ndarray, t: np.ndarray, H: int) -> np.ndarray:
    """
    Estimativa local por Transformada Discreta de Fourier (baseline clássico).

    Projeta o sinal medido na base de Fourier da janela (1 ciclo):
      - CC (x0)  = média do sinal;
      - Ac_i     = (2/m) * sum x[k] cos(2*pi*i*t[k]);
      - As_i     = (2/m) * sum x[k] sin(2*pi*i*t[k]).

    A TDF não modela decaimento, então `lambda` não é estimável (retorna NaN). É
    justamente isso que enviesa a estimativa da CC quando há decaimento exponencial.
    Os valores publicados de Rabêlo et al. (2011) são mantidos separadamente em
    `PUBLISHED_BASELINE`; esta função registra a comparação local sob a convenção
    de tempo normalizado deste script.
    """
    m = measured.size
    w0 = 2.0 * np.pi
    theta = np.full(2 + 2 * H, np.nan, dtype=float)
    theta[0] = float(np.mean(measured))  # x0 (CC)
    # theta[1] = lambda -> NaN (TDF não modela decaimento)
    for i in range(1, H + 1):
        ac = (2.0 / m) * float(np.sum(measured * np.cos(i * w0 * t)))
        as_ = (2.0 / m) * float(np.sum(measured * np.sin(i * w0 * t)))
        theta[2 + 2 * (i - 1)] = ac
        theta[2 + 2 * (i - 1) + 1] = as_
    return theta


def default_bounds(H: int, amp_bound: float = 1.5, lam_max: float = 5.0) -> np.ndarray:
    """
    Bounds de busca padrão, forma (2+2H, 2) com colunas [low, high].

    Amplitudes (x0 e Ac/As) em [-amp_bound, amp_bound]; lambda em [0, lam_max].
    """
    lo = np.full(2 + 2 * H, -amp_bound, dtype=float)
    hi = np.full(2 + 2 * H, amp_bound, dtype=float)
    lo[1] = 0.0  # lambda >= 0
    hi[1] = lam_max
    return np.stack([lo, hi], axis=1)


class ACOR:
    """
    ACO para domínios contínuos (SOCHA; DORIGO, 2008).

    Mantém um arquivo de `k` soluções ordenado por qualidade (menor RMSE primeiro). A
    cada iteração, gera `m` novas soluções amostrando kernels gaussianos centrados nas
    soluções do arquivo, junta tudo e mantém as `k` melhores.

    Parâmetros
    ----------
    signal : HarmonicSignal
        Instância usada para gerar o sinal sintético observado (ground truth).
    samples : int
        Nº de amostras na janela (1 ciclo). Padrão 64 (como no paper).
    k : int
        Tamanho do arquivo de soluções.
    ants : int
        Nº de soluções geradas por iteração (m).
    q : float
        Pressão de seleção / localidade. Menor => favorece o topo do arquivo.
    xi : float
        Velocidade de convergência (análogo à evaporação). Maior => busca mais larga.
    iterations : int
        Nº máximo de iterações.
    noise_snr_db : float | None
        SNR (dB) do ruído de medição adicionado ao sinal. None => sinal limpo.
    bounds : np.ndarray | None
        Bounds de busca (2+2H, 2). None => `default_bounds`.
    seed : int | None
        Semente aleatória (reprodutibilidade).
    early_stop_patience : int
        Nº de iterações sem melhora até interromper.
    early_stop_tol : float
        Melhora mínima de RMSE para resetar o contador de estagnação.
    out_dir : str
        Pasta para CSV/figuras/JSON.
    """

    def __init__(
        self,
        signal: HarmonicSignal,
        samples: int = 64,
        k: int = 50,
        ants: int = 30,
        q: float = 0.1,
        xi: float = 0.85,
        iterations: int = 1000,
        noise_snr_db: Optional[float] = None,
        bounds: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        early_stop_patience: int = 150,
        early_stop_tol: float = 1e-9,
        out_dir: str = "outputs",
    ):
        self.signal = signal
        self.H = signal.H
        self.dim = 2 + 2 * self.H
        self.samples = samples
        self.k = k
        self.ants = ants
        self.q = q
        self.xi = xi
        self.iterations = iterations
        self.noise_snr_db = noise_snr_db
        self.rng = np.random.default_rng(seed)
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol
        self.out_dir = out_dir

        self.bounds = bounds if bounds is not None else default_bounds(self.H)
        self.lo = self.bounds[:, 0]
        self.hi = self.bounds[:, 1]

        # --- SINAL MEDIDO (ground truth + ruído opcional) ---
        self.t = sample_times(samples)
        self.basis = harmonic_basis(self.t, self.H)
        self.measured = generate_signal(signal, self.t, noise_snr_db, self.rng)

        # --- PESOS DO ARQUIVO (gaussianos por ranking, fixos) ---
        ranks = np.arange(1, k + 1)
        self.weights = (1.0 / (q * k * np.sqrt(2.0 * np.pi))) * np.exp(
            -((ranks - 1) ** 2) / (2.0 * (q**2) * (k**2))
        )
        self.probs = self.weights / self.weights.sum()

        # --- ESTADO ---
        self.archive: np.ndarray = np.empty((k, self.dim), dtype=float)
        self.archive_rmse: np.ndarray = np.empty(k, dtype=float)
        self.history: List[Dict] = []
        self.global_best: Dict = {"theta": None, "rmse": np.inf, "i": 0}

    def _cost(self, thetas: np.ndarray) -> np.ndarray:
        """RMSE de um conjunto de candidatos (forma (P, dim) -> (P,))."""
        return rmse(thetas, self.t, self.basis, self.measured)

    def _init_archive(self) -> None:
        """Inicializa o arquivo com soluções uniformes nos bounds, ordenadas por RMSE."""
        samples = self.rng.uniform(
            self.lo, self.hi, size=(self.k, self.dim)
        )
        costs = self._cost(samples)
        order = np.argsort(costs)
        self.archive = samples[order]
        self.archive_rmse = costs[order]

    def _sample_ants(self) -> np.ndarray:
        """
        Gera `ants` novos candidatos a partir dos kernels gaussianos do arquivo.

        Para cada formiga: sorteia uma solução-guia l ~ probs; para cada dimensão d,
        amostra de N(mu = archive[l,d], sigma = xi * média_e |archive[e,d]-archive[l,d]|).
        """
        # sigma[l, d]: desvio por solução-guia e dimensão (espalhamento do arquivo)
        # |archive[e,d] - archive[l,d]| somado em e, média sobre (k-1)
        diffs = np.abs(
            self.archive[None, :, :] - self.archive[:, None, :]
        )  # (k, k, dim)
        sigma = self.xi * diffs.sum(axis=1) / max(1, self.k - 1)  # (k, dim)

        guides = self.rng.choice(self.k, size=self.ants, p=self.probs)  # (ants,)
        mu = self.archive[guides]  # (ants, dim)
        sd = sigma[guides]  # (ants, dim)
        new = self.rng.normal(mu, sd)  # (ants, dim)
        np.clip(new, self.lo, self.hi, out=new)
        return new

    def run(self, verbose: bool = True) -> Dict:
        """Executa o loop principal do ACOR e retorna a melhor solução global."""
        os.makedirs(self.out_dir, exist_ok=True)
        self._init_archive()
        no_improve = 0

        for it in range(1, self.iterations + 1):
            new = self._sample_ants()
            new_rmse = self._cost(new)

            # --- Atualização do arquivo: junta, ordena, mantém as k melhores ---
            all_theta = np.vstack([self.archive, new])
            all_rmse = np.concatenate([self.archive_rmse, new_rmse])
            order = np.argsort(all_rmse)[: self.k]
            self.archive = all_theta[order]
            self.archive_rmse = all_rmse[order]

            best_rmse = float(self.archive_rmse[0])

            if best_rmse < self.global_best["rmse"] - self.early_stop_tol:
                no_improve = 0
            else:
                no_improve += 1

            if best_rmse < self.global_best["rmse"]:
                self.global_best = {
                    "theta": self.archive[0].copy(),
                    "rmse": best_rmse,
                    "i": it,
                }

            self.history.append(
                {
                    "iteration": it,
                    "best_rmse": best_rmse,
                    "best_fitness": float(fitness(np.array(best_rmse))),
                    "archive_mean_rmse": float(self.archive_rmse.mean()),
                    "global_best_rmse": self.global_best["rmse"],
                }
            )

            if verbose and (it % max(1, self.iterations // 10) == 0 or it <= 3):
                print(
                    f"[{self.signal.name}] it={it:4d} | best_rmse={best_rmse:.3e} "
                    f"| FA={fitness(np.array(best_rmse)):.2f} "
                    f"| arquivo_médio={self.archive_rmse.mean():.3e}"
                )

            if no_improve >= self.early_stop_patience:
                if verbose:
                    print(
                        f"[EarlyStop] Sem melhora por {self.early_stop_patience} "
                        f"iterações em it={it}."
                    )
                break

        return self.global_best

    # ------------------------------------------------------------------ #
    # Métricas e validação
    # ------------------------------------------------------------------ #

    def parameter_errors(self, theta: np.ndarray) -> Dict:
        """
        Erro percentual por parâmetro vs. ground truth e erro médio.

        Segue o paper: erro (%) = |estimado - referência| / |referência| * 100. Lambda
        é incluído se estimável (não-NaN). Parâmetros com referência nula são ignorados
        no erro médio (evita divisão por zero), mas reportados individualmente.
        """
        true = self.signal.true_vector()
        names = ["x0", "lambda"]
        for i in range(1, self.H + 1):
            names += [f"Ac_{i}", f"As_{i}"]

        rows = []
        errs = []
        for idx, name in enumerate(names):
            ref = float(true[idx])
            est = float(theta[idx])
            if np.isnan(est):
                err = None
            elif abs(ref) < 1e-12:
                err = None
            else:
                err = abs(est - ref) / abs(ref) * 100.0
                errs.append(err)
            rows.append({"param": name, "ref": ref, "est": est, "err_pct": err})

        mean_err = float(np.mean(errs)) if errs else float("nan")
        return {"rows": rows, "mean_err_pct": mean_err}

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
        """Plota a curva de convergência (RMSE do melhor global, escala log)."""
        if not self.history:
            return
        iters = [h["iteration"] for h in self.history]
        gbest = [h["global_best_rmse"] for h in self.history]

        plt.figure(figsize=(9, 6))
        plt.semilogy(iters, gbest, label="Melhor global (RMSE)")
        plt.xlabel("Iteração")
        plt.ylabel("RMSE (escala log)")
        plt.title(f"Convergência ACOR — {self.signal.name}")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_reconstruction(self, path: Optional[str] = None) -> None:
        """
        Compara o sinal de referência, a reconstrução ACOR e a TDF (replica Figs. 3/4).

        Reamostra em uma grade fina para uma curva suave.
        """
        theta_aco = self.global_best["theta"]
        if theta_aco is None:
            return
        theta_tdf = tdf_estimate(self.measured, self.t, self.H)

        t_fine = sample_times(1000)
        basis_fine = harmonic_basis(t_fine, self.H)
        ref = reconstruct(self.signal.true_vector(), t_fine, basis_fine)
        aco = reconstruct(theta_aco, t_fine, basis_fine)
        # TDF: lambda é NaN; para reconstruir, trata CC como constante (lam=0)
        theta_tdf_plot = theta_tdf.copy()
        theta_tdf_plot[1] = 0.0
        tdf = reconstruct(theta_tdf_plot, t_fine, basis_fine)

        plt.figure(figsize=(11, 6))
        plt.plot(t_fine, ref, label="Referência", color="black", linewidth=2)
        plt.plot(t_fine, aco, label="ACOR", color="tab:red", linestyle="--")
        plt.plot(t_fine, tdf, label="TDF", color="tab:blue", linestyle=":")
        plt.scatter(
            self.t, self.measured, s=18, color="gray", alpha=0.6,
            label="Amostras sintéticas", zorder=5,
        )
        plt.xlabel("Tempo normalizado (1 ciclo)")
        plt.ylabel("Amplitude (pu)")
        plt.title(f"Reconstrução do sinal — {self.signal.name}")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=160, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_spectrum(self, path: Optional[str] = None) -> None:
        """Barras das amplitudes estimadas (ACOR e TDF) vs. referência, por harmônico."""
        theta_aco = self.global_best["theta"]
        if theta_aco is None:
            return
        theta_tdf = tdf_estimate(self.measured, self.t, self.H)
        true = self.signal.true_vector()

        # Magnitude por harmônico: sqrt(Ac^2 + As^2)
        def mags(theta: np.ndarray) -> np.ndarray:
            out = []
            for i in range(1, self.H + 1):
                ac = theta[2 + 2 * (i - 1)]
                as_ = theta[2 + 2 * (i - 1) + 1]
                out.append(np.hypot(ac, as_))
            return np.array(out)

        h = np.arange(1, self.H + 1)
        ref_m = mags(true)
        aco_m = mags(theta_aco)
        tdf_m = mags(theta_tdf)
        width = 0.27

        plt.figure(figsize=(11, 6))
        plt.bar(h - width, ref_m, width, label="Referência", color="black")
        plt.bar(h, aco_m, width, label="ACOR", color="tab:red")
        plt.bar(h + width, tdf_m, width, label="TDF", color="tab:blue")
        plt.xlabel("Ordem do harmônico")
        plt.ylabel("Magnitude  $\\sqrt{A_c^2 + A_s^2}$ (pu)")
        plt.title(f"Espectro estimado — {self.signal.name}")
        plt.xticks(h)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=160, bbox_inches="tight")
        else:
            plt.show()
        plt.close()


# Tensão — V(t) (RABÊLO et al., 2011, Eq. 7)
voltage = HarmonicSignal(
    name="Tensão",
    x0=0.0550,
    lam=0.4,
    harmonics=[
        (0.9829, 0.1842),  # h1 (fundamental)
        (0.0141, 0.0245),  # h2
        (0.0077, 0.0197),  # h3
        (0.0050, 0.0168),  # h4
        (0.0039, 0.0154),  # h5
        (0.0033, 0.0161),  # h6
        (0.0033, 0.0230),  # h7
    ],
)

# Corrente — I(t) (RABÊLO et al., 2011, Eq. 8)
current = HarmonicSignal(
    name="Corrente",
    x0=0.2491,
    lam=0.4,
    harmonics=[
        (0.95872, 0.2841),  # h1 (fundamental)
        (0.0619, 0.1054),  # h2
        (0.0329, 0.0811),  # h3
        (0.0206, 0.0643),  # h4
        (0.0146, 0.0528),  # h5
        (0.0116, 0.0448),  # h6
        (0.0052, 0.0401),  # h7
    ],
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ACOR para estimação de harmônicos (replicação de Rabêlo et al., 2011)."
    )
    parser.add_argument(
        "--signal",
        choices=["voltage", "current"],
        default="voltage",
        help="Sinal a estimar (tensão ou corrente).",
    )
    parser.add_argument(
        "--samples", type=int, default=64, help="Amostras na janela (1 ciclo)."
    )
    parser.add_argument(
        "--k", type=int, default=50, help="Tamanho do arquivo de soluções."
    )
    parser.add_argument(
        "--ants", type=int, default=30, help="Soluções geradas por iteração (m)."
    )
    parser.add_argument(
        "--q", type=float, default=0.1, help="Pressão de seleção / localidade."
    )
    parser.add_argument(
        "--xi", type=float, default=0.85, help="Velocidade de convergência (evaporação)."
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="Número máximo de iterações."
    )
    parser.add_argument(
        "--noise_snr",
        type=float,
        default=None,
        help="SNR do ruído de medição em dB (omitido => sinal limpo).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semente aleatória (reprodutibilidade)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Diretório de saída para CSV/figuras/JSON.",
    )
    args = parser.parse_args()

    signal = voltage if args.signal == "voltage" else current

    aco = ACOR(
        signal=signal,
        samples=args.samples,
        k=args.k,
        ants=args.ants,
        q=args.q,
        xi=args.xi,
        iterations=args.iters,
        noise_snr_db=args.noise_snr,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    best = aco.run(verbose=True)

    # --- Validação: erro por parâmetro (ACOR vs. TDF) ---
    aco_err = aco.parameter_errors(best["theta"])
    theta_tdf = tdf_estimate(aco.measured, aco.t, aco.H)
    tdf_err = aco.parameter_errors(theta_tdf)
    published = PUBLISHED_BASELINE[args.signal]

    print("\n=== Erro por parâmetro (%) ===")
    print(f"{'param':>8} | {'ref':>10} | {'ACOR est':>12} | {'ACOR %':>8} | {'TDF %':>8}")
    for a_row, t_row in zip(aco_err["rows"], tdf_err["rows"]):
        a_pct = f"{a_row['err_pct']:.4f}" if a_row["err_pct"] is not None else "—"
        t_pct = f"{t_row['err_pct']:.4f}" if t_row["err_pct"] is not None else "—"
        print(
            f"{a_row['param']:>8} | {a_row['ref']:>10.5f} | {a_row['est']:>12.5f} "
            f"| {a_pct:>8} | {t_pct:>8}"
        )
    print(
        f"\nErro médio — ACOR: {aco_err['mean_err_pct']:.4f}%  |  "
        f"TDF: {tdf_err['mean_err_pct']:.4f}%"
    )
    print(
        "Baseline publicado (Rabêlo et al., 2011) — "
        f"PSO: {published['pso_mean_err_pct']:.4f}% | "
        f"TDF: {published['tdf_mean_err_pct']:.4f}%"
    )
    print(
        "Nota: a TDF acima é a implementação local sob tempo normalizado; "
        "use o baseline publicado como referência bibliográfica."
    )

    # --- Artefatos ---
    base = os.path.join(args.out_dir, f"eletrica_{args.signal}")
    os.makedirs(args.out_dir, exist_ok=True)
    aco.save_history_csv(base + "_history.csv")
    aco.plot_convergence(base + "_convergence.png")
    aco.plot_reconstruction(base + "_reconstruction.png")
    aco.plot_spectrum(base + "_spectrum.png")

    meta = {
        "signal": signal.name,
        "data_source": DATA_SOURCE,
        "synthetic_benchmark": True,
        "time_convention": TIME_CONVENTION,
        "parameter_vector_order": PARAMETER_VECTOR_ORDER,
        "harmonics": signal.H,
        "samples": args.samples,
        "noise_snr_db": args.noise_snr,
        "best_rmse": best["rmse"],
        "best_iteration": best["i"],
        "best_theta": best["theta"].tolist() if best["theta"] is not None else None,
        "aco_param_errors": aco_err,
        "tdf_param_errors": tdf_err,
        "local_tdf_mean_err_pct": tdf_err["mean_err_pct"],
        "published_baseline_rabelo_2011": published,
    }
    with open(base + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nMelhor solução (RMSE={best['rmse']:.3e}) salva em {base}_*")
