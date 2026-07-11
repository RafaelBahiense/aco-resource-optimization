# ACOR para Estimação Harmônica

Este documento descreve o funcionamento de `src/aco_eletrica.py` pelo ponto de vista
do código. A explicação conceitual de Fourier, harmônicos e sinais elétricos fica em
`docs/conceitos_eletrica.md`.

---

## 1) Constantes e proveniência

O script começa registrando convenções usadas nos resultados e no JSON final:

- `TIME_CONVENTION` — informa que o tempo é normalizado: uma janela de 1 ciclo usa
  `t` em `[0, 1)`, com fundamental `cos(2*pi*t)`.
- `PARAMETER_VECTOR_ORDER` — fixa a ordem do vetor de decisão:

  ```text
  [x0, lambda, Ac_1, As_1, ..., Ac_H, As_H]
  ```

- `DATA_SOURCE` — descreve a origem dos sinais de tensão e corrente: Rabêlo et al.
  (2011), Eqs. 7 e 8.
- `PUBLISHED_BASELINE` — guarda os erros médios publicados para PSO e TDF. Esses
  valores são referência bibliográfica; o PSO não é reimplementado no script.

**Papel:** evitar que os artefatos gerados percam informação sobre origem dos dados,
ordem dos parâmetros e diferença entre baseline publicado e baseline local.

---

## 2) Modelo de dados

### `@dataclass HarmonicSignal`

Representa **uma instância do problema**:

- `name: str` — rótulo usado em logs, gráficos e metadados.
- `x0: float` — coeficiente da componente contínua inicial.
- `lam: float` — constante de decaimento exponencial.
- `harmonics: List[Tuple[float, float]]` — pares `(Ac_i, As_i)` do 1º ao H-ésimo
  harmônico.

### `HarmonicSignal.H`

Retorna:

```python
len(self.harmonics)
```

No benchmark atual, `H == 7` para tensão e corrente.

### `HarmonicSignal.true_vector() -> ndarray`

Monta o vetor verdadeiro na ordem usada por todo o algoritmo:

```text
[x0, lambda, Ac_1, As_1, Ac_2, As_2, ..., Ac_H, As_H]
```

**Invariante importante:** qualquer função que avalia, reconstrói, compara ou plota
soluções assume exatamente essa ordem.

---

## 3) Sinais embutidos

O arquivo define duas instâncias:

- `voltage = HarmonicSignal(...)` — forma de onda de tensão.
- `current = HarmonicSignal(...)` — forma de onda de corrente.

Ambas vêm dos casos de teste de Rabêlo et al. (2011) e têm:

- `x0`;
- `lambda`;
- sete pares `(Ac_i, As_i)`.

**Papel no código:** são o _ground truth_ sintético. O algoritmo gera o sinal observado
a partir desses parâmetros e depois tenta recuperá-los.

---

## 4) Base temporal e matriz harmônica

### `sample_times(samples, cycles=1.0) -> ndarray`

Cria os pontos de amostragem:

```python
np.linspace(0.0, cycles, samples, endpoint=False)
```

Com o padrão `samples=64` e `cycles=1.0`, o vetor cobre um ciclo sem repetir o início
do ciclo seguinte.

### `harmonic_basis(t, H) -> ndarray`

Monta uma matriz fixa `B` com forma:

```text
(len(t), 2H)
```

As colunas seguem a ordem:

```text
cos(1*2*pi*t), sin(1*2*pi*t),
cos(2*2*pi*t), sin(2*2*pi*t),
...
cos(H*2*pi*t), sin(H*2*pi*t)
```

**Motivação:** a parte harmônica pode ser calculada por multiplicação matricial, sem
recriar senos e cossenos a cada candidato.

---

## 5) Reconstrução e erro

### `reconstruct(theta, t, basis) -> ndarray`

Reconstrói o sinal a partir de um vetor candidato ou de vários candidatos.

Para um candidato:

```text
theta.shape == (2 + 2H,)
```

Para vários candidatos:

```text
theta.shape == (P, 2 + 2H)
```

A reconstrução é:

$$
\hat{x}(t) =
x_0 e^{-\lambda t}
+ \sum_{i=1}^{H}
\left[
A_{c_i}\cos(2\pi i t)
+ A_{s_i}\sin(2\pi i t)
\right].
$$

No código:

- `x0 = th[:, 0:1]`
- `lam = th[:, 1:2]`
- `amps = th[:, 2:]`
- `dc = x0 * np.exp(-lam * t[None, :])`
- `harm = amps @ basis.T`
- `recon = dc + harm`

### `rmse(theta, t, basis, measured) -> ndarray`

Calcula o erro de reconstrução:

$$
\mathrm{RMSE} =
\sqrt{
\frac{1}{m}
\sum_{j=1}^{m}
\left(\hat{x}(t_j)-x(t_j)\right)^2
}.
$$

É vetorizado: se `theta` contém `P` candidatos, retorna `P` valores de RMSE.

### `fitness(rmse_val, delta=1e-5) -> ndarray`

Converte RMSE em aptidão:

$$
\mathrm{FA} = \frac{1}{\mathrm{RMSE} + \Delta},
\qquad \Delta = 10^{-5}.
$$

**Convenção:** o ACOR minimiza RMSE, mas o relatório também imprime `FA`, em que maior
é melhor.

---

## 6) Geração do sinal observado

### `generate_signal(signal, t, noise_snr_db=None, rng=None) -> ndarray`

Gera o sinal observado a partir do vetor verdadeiro:

1. Calcula `basis = harmonic_basis(t, H)`.
2. Calcula `clean = reconstruct(signal.true_vector(), t, basis)`.
3. Se `noise_snr_db is None`, retorna o sinal limpo.
4. Se `noise_snr_db` for informado, adiciona ruído gaussiano com a SNR desejada.

O ruído é opcional e serve para experimentos de robustez. O baseline limpo usa
`noise_snr_db=None`.

---

## 7) Baseline local por TDF

### `tdf_estimate(measured, t, H) -> ndarray`

Calcula uma estimativa local por projeção na base de Fourier:

- `x0` recebe a média do sinal;
- `lambda` recebe `NaN`, pois a TDF local não estima decaimento exponencial;
- cada par `(Ac_i, As_i)` é estimado por projeção:

  $$
  A_{c_i} = \frac{2}{m}\sum_{j=1}^{m}x(t_j)\cos(2\pi i t_j),
  $$

  $$
  A_{s_i} = \frac{2}{m}\sum_{j=1}^{m}x(t_j)\sin(2\pi i t_j).
  $$

**Separação importante:** essa TDF é a implementação local sob a convenção de tempo do
script. Os valores de TDF publicados por Rabêlo et al. (2011) ficam em
`PUBLISHED_BASELINE`.

---

## 8) Bounds do espaço de busca

### `default_bounds(H, amp_bound=1.5, lam_max=5.0) -> ndarray`

Cria uma matriz com forma:

```text
(2 + 2H, 2)
```

As colunas são `[low, high]`.

Regras padrão:

- `x0` e todos os coeficientes `Ac_i`, `As_i` ficam em `[-amp_bound, amp_bound]`.
- `lambda` fica em `[0, lam_max]`.

No ACOR, esses limites são guardados em:

- `self.lo = self.bounds[:, 0]`
- `self.hi = self.bounds[:, 1]`

Durante a amostragem, novos candidatos são projetados para o domínio viável por:

```python
np.clip(new, self.lo, self.hi, out=new)
```

Isso impede que uma formiga continue com parâmetro fora dos limites definidos.

---

## 9) Núcleo ACOR — `class ACOR`

### 9.1 Estado interno e hiperparâmetros

O construtor recebe:

- `signal` — instância `HarmonicSignal`.
- `samples` — número de amostras na janela; padrão `64`.
- `k` — tamanho do arquivo de soluções; padrão `50`.
- `ants` — novas soluções geradas por iteração; padrão `30`.
- `q` — pressão de seleção; menor valor concentra escolha nas melhores soluções.
- `xi` — escala dos desvios dos kernels gaussianos.
- `iterations` — máximo de iterações.
- `noise_snr_db` — ruído opcional.
- `bounds` — limites por dimensão; se omitido, usa `default_bounds`.
- `seed` — semente do gerador aleatório.
- `early_stop_patience` — iterações sem melhora antes de parar.
- `early_stop_tol` — melhora mínima de RMSE para contar como avanço.
- `out_dir` — diretório de saída.

Durante a inicialização:

- `self.dim = 2 + 2*self.H`
- `self.t = sample_times(samples)`
- `self.basis = harmonic_basis(self.t, self.H)`
- `self.measured = generate_signal(...)`
- `self.archive` guarda `k` vetores candidatos.
- `self.archive_rmse` guarda o RMSE de cada vetor do arquivo.
- `self.global_best` guarda a melhor solução global.

Aqui, **arquivo** significa _arquivo de soluções_ (_solution archive_), uma memória em
RAM com os melhores candidatos, não um arquivo `.csv` ou `.json` em disco.

---

### 9.2 Pesos de seleção do arquivo

O ACOR ordena o arquivo do menor para o maior RMSE. Depois atribui pesos por ranking:

$$
w_l =
\frac{1}{qk\sqrt{2\pi}}
\exp\left(
-\frac{(l-1)^2}{2q^2k^2}
\right),
$$

para `l = 1, ..., k`.

As probabilidades são:

$$
p_l = \frac{w_l}{\sum_{r=1}^{k} w_r}.
$$

No código:

```python
ranks = np.arange(1, k + 1)
self.weights = ...
self.probs = self.weights / self.weights.sum()
```

**Efeito de `q`:**

- `q` menor aumenta a preferência pelos primeiros colocados.
- `q` maior distribui melhor a chance entre mais soluções do arquivo.

---

### 9.3 Inicialização do arquivo

#### `_init_archive() -> None`

1. Sorteia `k` candidatos uniformemente entre `self.lo` e `self.hi`.
2. Calcula o RMSE de todos com `_cost(samples)`.
3. Ordena pelo RMSE crescente.
4. Guarda os `k` candidatos em `self.archive`.

Trecho central:

```python
samples = self.rng.uniform(self.lo, self.hi, size=(self.k, self.dim))
costs = self._cost(samples)
order = np.argsort(costs)
self.archive = samples[order]
self.archive_rmse = costs[order]
```

---

### 9.4 Amostragem por kernels gaussianos

#### `_sample_ants() -> ndarray`

Gera `ants` novos candidatos.

Para cada solução `l` do arquivo e dimensão `d`, o desvio é calculado por:

$$
\sigma_{l,d}
=
\xi
\frac{
\sum_{e=1}^{k}
\left|
\theta_{e,d} - \theta_{l,d}
\right|
}{
k-1
}.
$$

Depois cada formiga:

1. Sorteia uma solução-guia `l` com probabilidade `self.probs`.
2. Usa `mu = self.archive[l]`.
3. Usa `sd = sigma[l]`.
4. Gera um novo vetor por distribuição normal:

   $$
   \theta'_d \sim \mathcal{N}(\mu_d, \sigma_d).
   $$

5. Aplica `clip` nos bounds.

No código:

```python
guides = self.rng.choice(self.k, size=self.ants, p=self.probs)
mu = self.archive[guides]
sd = sigma[guides]
new = self.rng.normal(mu, sd)
np.clip(new, self.lo, self.hi, out=new)
```

**Papel de `xi`:**

- `xi` maior aumenta o espalhamento da busca.
- `xi` menor torna a busca mais concentrada ao redor das soluções do arquivo.

---

### 9.5 Loop de otimização

#### `run(verbose=True) -> Dict`

Executa o ACOR:

1. Cria `out_dir`.
2. Inicializa o arquivo com `_init_archive()`.
3. Para cada iteração:
   - gera novas soluções com `_sample_ants()`;
   - calcula `new_rmse`;
   - junta arquivo antigo e novas soluções;
   - ordena por RMSE;
   - mantém apenas as `k` melhores;
   - atualiza `global_best`;
   - registra métricas em `history`;
   - imprime progresso se `verbose=True`;
   - interrompe se `no_improve >= early_stop_patience`.

A atualização principal do arquivo é:

```python
all_theta = np.vstack([self.archive, new])
all_rmse = np.concatenate([self.archive_rmse, new_rmse])
order = np.argsort(all_rmse)[: self.k]
self.archive = all_theta[order]
self.archive_rmse = all_rmse[order]
```

**Diferença para `aco_agro.py`:** aqui não existe matriz `tau` em uma grade. O arquivo
de soluções e os kernels gaussianos cumprem o papel de concentrar a busca.

---

## 10) Métricas e validação

### `parameter_errors(theta) -> Dict`

Compara um vetor estimado com `signal.true_vector()`:

$$
E_j(\%) =
\frac{
\left|\hat{\theta}_j - \theta_j\right|
}{
\left|\theta_j\right|
}
\times 100.
$$

Regras:

- `lambda` entra no erro se o valor estimado não for `NaN`.
- parâmetros com referência zero são ignorados no erro médio para evitar divisão por
  zero.
- a saída contém `rows` e `mean_err_pct`.

### `_cost(thetas) -> ndarray`

Atalho interno para:

```python
rmse(thetas, self.t, self.basis, self.measured)
```

---

## 11) Artefatos

### `save_history_csv(path)`

Salva `self.history` em CSV. As colunas são:

- `iteration`
- `best_rmse`
- `best_fitness`
- `archive_mean_rmse`
- `global_best_rmse`

### `plot_convergence(path)`

Gera a curva de convergência em escala logarítmica do RMSE.

### `plot_reconstruction(path)`

Compara:

- sinal de referência;
- reconstrução ACOR;
- reconstrução TDF local;
- amostras sintéticas usadas na estimação.

Como a TDF retorna `NaN` para `lambda`, o gráfico usa `lambda = 0.0` apenas para
reconstruir a curva da TDF.

### `plot_spectrum(path)`

Plota, por harmônico, a magnitude:

$$
\sqrt{A_c^2 + A_s^2}.
$$

Compara referência, ACOR e TDF local.

---

## 12) CLI

Execução padrão para tensão:

```bash
pdm run python src/aco_eletrica.py --signal voltage
```

Execução para corrente com mais iterações:

```bash
pdm run python src/aco_eletrica.py --signal current --iters 3000
```

Experimento com ruído:

```bash
pdm run python src/aco_eletrica.py --signal voltage --noise_snr 30
```

Principais opções:

- `--signal {voltage,current}` — seleciona o sinal.
- `--samples` — amostras na janela.
- `--k` — tamanho do arquivo de soluções.
- `--ants` — novas soluções por iteração.
- `--q` — pressão de seleção.
- `--xi` — escala dos kernels gaussianos.
- `--iters` — máximo de iterações.
- `--noise_snr` — SNR do ruído em dB.
- `--seed` — semente aleatória.
- `--out_dir` — diretório dos artefatos.

---

## 13) Saídas geradas

Para:

```bash
pdm run python src/aco_eletrica.py --signal voltage --out_dir outputs
```

o prefixo é:

```text
outputs/eletrica_voltage_*
```

Arquivos esperados:

- `eletrica_voltage_history.csv`
- `eletrica_voltage_convergence.png`
- `eletrica_voltage_reconstruction.png`
- `eletrica_voltage_spectrum.png`
- `eletrica_voltage_meta.json`

Para corrente, o prefixo muda para:

```text
outputs/eletrica_current_*
```

O JSON inclui:

- sinal escolhido;
- fonte dos dados;
- convenção de tempo;
- ordem do vetor de parâmetros;
- número de harmônicos;
- melhor RMSE;
- melhor vetor estimado;
- erros por parâmetro do ACOR;
- erros por parâmetro da TDF local;
- baseline publicado de Rabêlo et al. (2011).

---

## 14) Invariantes e cuidados

- A ordem do vetor de parâmetros deve permanecer:

  ```text
  [x0, lambda, Ac_1, As_1, ..., Ac_H, As_H]
  ```

- `basis` usa colunas em pares cosseno/seno; mudar essa ordem quebra reconstrução,
  erros e espectro.
- O tempo é normalizado. Comparações com artigos ou implementações em segundos devem
  explicitar essa convenção.
- O ACOR minimiza RMSE; `fitness` é apenas uma transformação para relatório.
- `PUBLISHED_BASELINE` não é recalculado. Ele guarda valores publicados.
- A TDF local não estima `lambda`; por isso `lambda = NaN` aparece nos erros da TDF.
- O `clip` em `_sample_ants()` é parte da definição do espaço viável.
- `seed` fixa o gerador NumPy, mas mudar `k`, `ants`, `iters`, bounds ou ruído muda a
  trajetória da busca.
- O script trabalha com sinais sintéticos embutidos. Para usar medições externas, é
  necessário adaptar a entrada de `measured` e definir como obter ou validar o vetor de
  referência.
