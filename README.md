# ACO Codes

Implementações baseadas em **Ant Colony Optimization (ACO)** aplicadas a problemas de otimização contínua e agroeconômica, com ênfase em **Agricultura 4.0**.

O repositório contém duas aplicações principais:

1. `aco_x2.py` — **Minimização da função f(x)=x²** (exemplo didático)
2. `aco_agro.py` — **Otimização de insumos hídricos e nitrogenados** em culturas agrícolas (melão e alface-americana)

---

## Instalação

### Requisitos

- Python **3.12+**
- [PDM](https://pdm-project.org) (gerenciador de pacotes Python)

### Passos

```bash
git clone https://github.com/RafaelBahiense/aco_codes.git
cd aco-codes
pdm install
```

---

## Estrutura geral

```
src/
 ├── aco_x2.py      # ACO 1D para f(x)=x²
 ├── aco_agro.py    # ACO 2D para funções de produção agrícolas
docs/
 └── ...            # Documentação técnica detalhada
outputs/
 ├── *.csv          # Histórico das iterações
 ├── *.png          # Gráficos de convergência / heatmap / superfície 3D
 └── *.json         # Metadados com melhor solução e ótimo analítico
```

---

## `aco_x2.py`: ACO 1D (Função Quadrática)

Script simples para validar o comportamento do ACO na minimização de uma função contínua:

$$
f(x) = x^2
$$

### Uso via CLI

```bash
pdm run x2 --iterations 150 --plot --verbose
```

| Flag                     | Descrição                   | Padrão                   |
| ------------------------ | --------------------------- | ------------------------ |
| `--iterations`           | Iterações                   | `100`                    |
| `--n_ants`               | Formigas                    | `50`                     |
| `--evaporation_rate`     | Taxa de evaporação          | `0.9`                    |
| `--early_stop_threshold` | Parada antecipada           | `1e-5`                   |
| `--upper_bound`          | Limite superior             | `10.0`                   |
| `--lower_bound`          | Limite inferior             | `-10.0`                  |
| `--plot`                 | Exibe gráfico               | `False`                  |
| `--save_plot`            | Caminho para salvar gráfico | `plots/aco_progress.png` |

---

## `aco_agro.py`: ACO 2D (Agricultura 4.0)

Algoritmo de **colônia de formigas 2D** aplicado à otimização de **água (w)** e **nitrogênio (n)** em culturas agrícolas.

### Modelos disponíveis

| Cultura              | Função de produção (rendimento `y`)                              | Intervalos (w, n)           |
| -------------------- | ---------------------------------------------------------------- | --------------------------- |
| **Meloeiro**         | (y = 34.16737n + 70.77509w - 0.05781w^2 - 0.07612n^2)            | (0–700 mm, 0–350 kg N/ha)   |
| **Alface-americana** | (y = -12.49 + 388.1w - 6.02n - 1.042w^2 - 0.04563n^2 + 0.1564wn) | (0–250 mm, 100–240 kg N/ha) |

### Função-objetivo

$$
R(w,n) = p_y \cdot y(w,n) - c_w \cdot w - c_n \cdot n
$$

onde:

- (p_y) é o preço do produto (R$/kg),
- (c_w) e (c_n) são custos unitários de água e nitrogênio.

---

## Uso via CLI

```bash
pdm run agro --crop melon --objective revenue
```

| Flag               | Descrição                                              | Padrão     |
| ------------------ | ------------------------------------------------------ | ---------- |
| `--crop`           | `lettuce` ou `melon`                                   | `melon`    |
| `--objective`      | `revenue` (receita líquida) ou `yield` (produtividade) | `revenue`  |
| `--iters`          | Iterações                                              | `1000`     |
| `--ants`           | Formigas por iteração                                  | `200`      |
| `--alpha`          | Peso do feromônio (τ^α)                                | `1.0`      |
| `--beta`           | Peso da heurística (η^β)                               | `1.0`      |
| `--rho`            | Evaporação (0–1)                                       | `0.1`      |
| `--q`              | Escala do depósito                                     | `1.0`      |
| `--elitist_weight` | Reforço no melhor global                               | `2.0`      |
| `--topk_fraction`  | Fração dos melhores que depositam                      | `0.2`      |
| `--seed`           | Semente aleatória                                      | `42`       |
| `--out_dir`        | Diretório de saída                                     | `outputs/` |

---

## Exemplos de execução

### Meloeiro — receita líquida

```bash
pdm run agro --crop melon --objective revenue --iters 1000 --ants 200
```

### Meloeiro — produtividade

```bash
pdm run agro --crop melon --objective yield
```

### Alface-americana — receita líquida

```bash
pdm run agro --crop lettuce --objective revenue
```

---

## Saídas geradas

A cada execução, o ACO gera automaticamente:

| Tipo | Arquivo             | Descrição                                  |
| ---- | ------------------- | ------------------------------------------ |
| CSV  | `*_history.csv`     | Histórico por iteração                     |
| PNG  | `*_convergence.png` | Curva de convergência                      |
| PNG  | `*_heatmap.png`     | Mapa de contorno da função                 |
| PNG  | `*_surface3d.png`   | Superfície 3D com colormap                 |
| JSON | `*_meta.json`       | Melhor solução e ótimo analítico projetado |

---
