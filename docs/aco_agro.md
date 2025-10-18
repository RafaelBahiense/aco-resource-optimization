# ACO em Grade 2D para Agricultura

## 1) Modelo de dados

### `@dataclass Crop`

Representa **uma instância do problema** (cultura):

- `name: str` — rótulo (ex.: “Meloeiro”, “Alface-americana”).
- `w_bounds: (float, float)` — domínio de água $w \in [w_{\min}, w_{\max}]$ (mm).
- `n_bounds: (float, float)` — domínio de nitrogênio $n \in [n_{\min}, n_{\max}]$ (kg/ha).
- `coeffs: (a0, a1, a2, a3, a4, a5)` — coeficientes da produção quadrática:

  $$
  y(w,n) = a_0 + a_1 w + a_2 n + a_3 w^2 + a_4 n^2 + a_5 w n.
  $$

- `price_py, cost_cw, cost_cn` — preço do produto (R$/kg) e custos variáveis: água (R$/(mm·ha)) e N (R$/kg).

**Papel:** encapsula parâmetros econômicos/agronômicos; toda avaliação $y$ e $R$ parte daqui.

---

## 2) Funções de avaliação

### `quadratic_yield(coeffs, w, n) -> ndarray`

- Calcula $y(w,n)$ (produtividade) de forma vetorizada sobre arrays (escalares, vetores, grades).

### `revenue(crop, w, n) -> ndarray`

- Calcula a receita líquida:

  $$
  R(w,n) = p_y \cdot y(w,n) \;-\; c_w \cdot w \;-\; c_n \cdot n,
  $$

  onde $y$ vem de `quadratic_yield`.

---

## 3) Discretização do espaço de decisão

### `make_grid(bounds, steps) -> ndarray`

- Cria uma grade 1D uniforme (linspace fechado) para cada eixo:
  - `W_vals = make_grid(crop.w_bounds, steps_w)`
  - `N_vals = make_grid(crop.n_bounds, steps_n)`
- A grade 2D vem de:
  - `W_grid, N_grid = np.meshgrid(W_vals, N_vals)`
  - `W_grid.shape == N_grid.shape == (steps_n, steps_w)`

**Motivação:** a ACO aqui é discreta, cada célula $(i,j)$ da grade representa um par candidato $(w_i, n_j)$.

---

## 4) Restrições genéricas

### `class Constraint`

- Modelo: $g(w,n) \le 0$ é factível; se $g(w,n) > 0$, aplica-se penalidade:

  $$
  \mathrm{pen}(w,n) = \max\!\bigl(0,\, g(w,n)\bigr)\cdot \text{weight}.
  $$

- Exemplo (teto de orçamento): $g(w,n) = c_w\, w + c_n\, n - B$.

**Integração (duas camadas):**

1. **Heurística (`eta_raw`)** recebe penalidade (reduz probabilidade de amostrar regiões inviáveis).
2. **Score** `_score_point` também subtrai penalidade (punição no ranking final).

---

## 5) Núcleo ACO — `class ACO2D`

### 5.1 Estado interno e hiperparâmetros

- **Objetivo:** `objective ∈ {"revenue","yield"}` decide se otimiza $R$ ou $y$.
- **Grade:** `W_vals`, `N_vals`, `W_grid`, `N_grid`.
- **Heurística `eta` $\in (0,1]$:**
  - `eta_raw = revenue(...)` se objetivo for receita liquida; `eta_raw = quadratic_yield(...)` se produtividade.
  - Penalidades de restrição são **subtraídas** de `eta_raw` antes da normalização.
  - Normalização min–max para $(0,1]$ com $\varepsilon$ pequeno:

    $$
    \eta \;=\; \frac{\eta_{\text{raw}} - \min(\eta_{\text{raw}})}{\max(\eta_{\text{raw}}) - \min(\eta_{\text{raw}})}\cdot (1-\varepsilon) + \varepsilon.
    $$

- **Feromônio `tau`:** matriz inicial de uns (sem viés).
- **RNG:** `np.random.default_rng(seed)` (reprodutibilidade).

**Hiperparâmetros de aprendizado:**

- `alpha` — peso do feromônio: $\tau^\alpha$.
- `beta` — peso da heurística: $\eta^\beta$.
- `rho` — evaporação (fração perdida por iteração).
- `q` — escala do depósito.
- `elitist_weight` — reforço extra no **melhor global** a cada iteração.
- `topk_fraction` — somente os **top-k%** da iteração depositam.
- `early_stop_patience` — encerra se não houver melhora após X iterações.

---

### 5.2 Avaliação de um ponto

#### `_score_point(w, n) -> float`

- **Base do objetivo:** $f(w,n) = R(w,n)$ **ou** $y(w,n)$, conforme `objective`.
- **Penalidade:** subtrai $\sum_k \max\!\bigl(0, g_k(w,n)\bigr) \cdot \text{weight}_k$.
- **Retorno:** `score` escalar a **maximizar** (quanto maior, melhor).

---

### 5.3 Loop de otimização

#### `run(verbose=True) -> Dict`

Itera $t = 1,\dots,T$:

1. **Probabilidades de amostragem** por célula $(i,j)$:

   $$
   \pi_{ij} \;\propto\; \bigl(\tau_{ij}\bigr)^{\alpha}\cdot \bigl(\eta_{ij}\bigr)^{\beta},
   \qquad \sum_{i,j} \pi_{ij} = 1.
   $$

   (Se degenerado, usa distribuição **uniforme**.)

2. **Amostragem (formigas):**
   - Sorteia `ants` índices com reposição via $\pi$.
   - Converte índice linear em $(i,j)$ e obtém $(w_i, n_j)$.

3. **Avaliação:**
   - `scores[k] = _score_point(w_k, n_k)` para cada formiga.
   - Determina **melhor da iteração** ($k=\arg\max$ dos scores).

4. **Melhor global:**
   - Se o melhor da iteração superou `best_score` (margem > $10^{-15}$), atualiza `global_best`, senão incrementa `no_improve`.

5. **Evaporação (global):**

   $$
   \tau \;\leftarrow\; (1-\rho)\cdot \tau.
   $$

6. **Depósito top-k:**
   - Seleciona `topk` índices com maiores scores.
   - Normaliza $[s_{\min}, s_{\max}]$ para $[0,1]$ e deposita, por célula visitada:

     $$
     \tau_{ij} \;\leftarrow\; \tau_{ij} \;+\; q \cdot \frac{s_k - s_{\min}}{\,s_{\max} - s_{\min}\,}.
     $$

7. **Reforço elitista (global best):**
   - Identifica a célula mais próxima de $(w^*, n^*)$ do **melhor global** e soma:

     $$
     \tau_{i^*j^*} \;\leftarrow\; \tau_{i^*j^*} \;+\; \text{elitist\_weight}\cdot q.
     $$

8. **Estabilidade numérica:** `tau = clip(tau, 1e-6, 1e6)`.

9. **Histórico:** registra métricas da iteração e do melhor global.

10. **Parada antecipada:** se `no_improve >= early_stop_patience`, encerra.

**Efeito combinado:** $\tau$ **concentra probabilidade** nas regiões boas; $\rho$ **evita congelamento prematuro**; $\eta$ atrai para áreas promissoras (**exploração guiada**).

---

## 6) Hiperparâmetros

- **`alpha`** ↑ → mais **intensificação** (seguir feromônio). Muito alto pode reduzir exploração.
- **`beta`** ↑ → mais confiança na **heurística** (mapa do objetivo). Útil no início.
- **`rho`** em $[0.05, 0.20]$ é um bom ponto de partida.  
  Alto demais → instável; baixo demais → estagnação.
- **`topk_fraction`** em $[0.10, 0.30]$ equilibra seletividade/diversidade.
- **`elitist_weight`** moderado (1–3) reforça o melhor global.

---

## 7) Objetivo “yield” vs “revenue”

- **Heurística `eta`:**
  - `yield` → favorece regiões de alta produtividade ($y$).
  - `revenue` → considera preço e custos ($R$), podendo mover o ótimo para menor insumo se a margem melhora.
- **Score:** mesma lógica, apenas troca a função base $y \leftrightarrow R$.

**Consequência:** o par ótimo $(w,n)$ pode **mudar** ao alternar o objetivo.

---

## 8) Robustez e fronteiras

- **Domínio:** escolhas **sempre** dentro de `w_bounds × n_bounds` (grade limita).
- **Restrições:** penalização dupla (heurística + score) desestimula e pune violações.
- **Reprodutibilidade:** `seed` fixa a sequência de amostras (RNG).
