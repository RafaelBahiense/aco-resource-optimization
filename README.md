# ACO Codes

Implementações baseadas em **Ant Colony Optimization (ACO)** aplicadas a problemas de otimização contínua, agricultura 4.0 e qualidade da energia elétrica.

O repositório contém três aplicações principais:

1. `aco_x2.py` — **Minimização da função f(x)=x²** (exemplo didático)
2. `aco_agro.py` — **Otimização de insumos hídricos e nitrogenados** em culturas agrícolas (melão e alface-americana)
3. `aco_eletrica.py` — **Estimação harmônica em sinais elétricos** com ACO contínuo (ACOR)

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
 ├── aco_eletrica.py # ACOR para estimação de harmônicos elétricos
docs/
 └── ...            # Documentação técnica detalhada
outputs/
 ├── *.csv          # Histórico das iterações
 ├── *.png          # Gráficos de convergência / reconstrução / espectro / superfícies
 └── *.json         # Metadados com melhor solução e informações do experimento
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

## `aco_eletrica.py`: ACOR para Estimação Harmônica

Algoritmo de **ACO para domínios contínuos (ACOR)** aplicado à estimação dos parâmetros harmônicos de sinais elétricos de tensão e corrente. O código replica os cenários sintéticos de Rabêlo et al. (2011), com componente contínua (CC) de decaimento exponencial e 7 harmônicos.

### Modelo do sinal

$$
x(t) = x_0 e^{-\lambda t} + \sum_{k=1}^{N_h}\left[a_k\cos(k\omega_0 t) + b_k\sin(k\omega_0 t)\right]
$$

O vetor estimado pelo ACOR segue a ordem:

```text
[x0, lambda, Ac_1, As_1, ..., Ac_7, As_7]
```

### Sinais disponíveis

| Sinal     | Origem                 | Descrição                                      |
| --------- | ---------------------- | ---------------------------------------------- |
| `voltage` | Rabêlo et al. (2011)   | Forma de onda sintética de tensão com 7 harmônicos |
| `current` | Rabêlo et al. (2011)   | Forma de onda sintética de corrente com 7 harmônicos |

### Uso via CLI

```bash
pdm run python src/aco_eletrica.py --signal voltage --samples 64 --iters 3000
```

```bash
pdm run python src/aco_eletrica.py --signal current --samples 64 --iters 3000
```

| Flag          | Descrição                                           | Padrão     |
| ------------- | --------------------------------------------------- | ---------- |
| `--signal`    | `voltage` (tensão) ou `current` (corrente)          | `voltage`  |
| `--samples`   | Amostras na janela de 1 ciclo                       | `64`       |
| `--k`         | Tamanho da memória de soluções do ACOR              | `50`       |
| `--ants`      | Soluções geradas por iteração                       | `30`       |
| `--q`         | Pressão de seleção / localidade                     | `0.1`      |
| `--xi`        | Velocidade de convergência (espalhamento da busca)  | `0.85`     |
| `--iters`     | Número máximo de iterações                          | `1000`     |
| `--noise_snr` | SNR do ruído de medição em dB                       | `None`     |
| `--seed`      | Semente aleatória                                   | `42`       |
| `--out_dir`   | Diretório de saída                                  | `outputs/` |

### Exemplos de execução

#### Tensão — cenário comparável ao Rabêlo et al. (2011)

```bash
pdm run python src/aco_eletrica.py --signal voltage --samples 64 --iters 3000 --seed 42 --out_dir outputs
```

#### Corrente — cenário comparável ao Rabêlo et al. (2011)

```bash
pdm run python src/aco_eletrica.py --signal current --samples 64 --iters 3000 --seed 42 --out_dir outputs
```

#### Tensão com ruído gaussiano

```bash
pdm run python src/aco_eletrica.py --signal voltage --samples 64 --iters 3000 --noise_snr 30
```

### Comparação com Rabêlo et al. (2011)

O código calcula:

- resultado do **ACOR proposto**;
- resultado da **TDF local**;
- erro percentual por parâmetro;
- RMSE da reconstrução.

Os valores publicados por Rabêlo et al. (2011) para PSO e TDF são mantidos como baseline bibliográfico no arquivo `*_meta.json`. O PSO não é reimplementado neste script.

### Saídas geradas

Cada execução gera arquivos com prefixo `eletrica_{signal}_` em `--out_dir`:

| Tipo | Arquivo                 | Descrição                                      |
| ---- | ----------------------- | ---------------------------------------------- |
| CSV  | `*_history.csv`         | Histórico de convergência por iteração         |
| PNG  | `*_convergence.png`     | Curva de convergência do RMSE                  |
| PNG  | `*_reconstruction.png`  | Sinal de referência × ACOR × TDF               |
| PNG  | `*_spectrum.png`        | Magnitudes harmônicas estimadas                |
| JSON | `*_meta.json`           | Melhor solução, erros, baseline e metadados    |

---

## Saídas geradas

As aplicações geram automaticamente arquivos em `outputs/` ou no diretório definido por `--out_dir`:

| Tipo | Arquivo             | Descrição                                  |
| ---- | ------------------- | ------------------------------------------ |
| CSV  | `*_history.csv`     | Histórico por iteração                     |
| PNG  | `*_convergence.png` | Curva de convergência                      |
| PNG  | `*_heatmap.png`     | Mapa de contorno da função agrícola        |
| PNG  | `*_surface3d.png`   | Superfície 3D agrícola                     |
| PNG  | `*_reconstruction.png` | Reconstrução do sinal elétrico          |
| PNG  | `*_spectrum.png`    | Espectro harmônico estimado                |
| JSON | `*_meta.json`       | Metadados e melhor solução                 |

---
