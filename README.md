# ACO Codes

Algoritmos baseados em **Ant Colony Optimization (ACO)** para resolver problemas de otimização:

1. **Minimização da função quadrática f(x) = x²** (`aco_x2.py`)
2. **Otimização da produção do meloeiro** (`aco_agro.py`)

---

## Requisitos

- Python `3.12`
- [PDM](https://pdm-project.org) (gerenciador de pacotes Python)

## Instalação

1. Clone o repositório:

```bash
git clone
cd aco-codes
```

2. Instale as dependências com PDM:

```bash
pdm install
```

---

## `aco_x2.py`: ACO para minimizar f(x) = x²

Esse script aplica ACO para encontrar o mínimo da função:

```text
f(x) = x²
```

### Uso via CLI:

```bash
python src/aco_x2.py [opções]
```

#### Opções disponíveis:

| Flag                     | Descrição                     | Padrão                   |
| ------------------------ | ----------------------------- | ------------------------ |
| `--iterations`           | Número de iterações           | `100`                    |
| `--n_ants`               | Número de formigas            | `50`                     |
| `--evaporation_rate`     | Taxa de evaporação            | `0.9`                    |
| `--early_stop_threshold` | Critério de parada antecipada | `1e-5`                   |
| `--upper_bound`          | Limite superior do domínio    | `10.0`                   |
| `--lower_bound`          | Limite inferior do domínio    | `-10.0`                  |
| `--plot`                 | Mostrar gráfico               | `False`                  |
| `--save_plot CAMINHO`    | Caminho para salvar gráfico   | `plots/aco_progress.png` |

#### Exemplo:

```bash
pdm run x2 --iterations 150 --plot --verbose
```

---

## `aco_agro.py`: ACO para otimização da produção do meloeiro

Simula o cultivo do meloeiro ajustando:

- **Lâmina de água (w)** em mm
- **Dose de nitrogênio (n)** em kg/ha

### Função de produção

```text
Yield(w, n) = 34.16737 * n + 70.77509 * w - 0.05781 * w² - 0.07612 * n²
```

### Uso via CLI:

```bash
pdm run agro [opções]
```

#### Opções disponíveis:

| Flag                  | Descrição                       | Padrão                   |
| --------------------- | ------------------------------- | ------------------------ |
| `--iterations`        | Número de iterações             | `200`                    |
| `--n_ants`            | Número de formigas por iteração | `50`                     |
| `--evaporation_rate`  | Taxa de evaporação              | `0.9`                    |
| `--verbose`           | Exibir logs detalhados (`INFO`) | `False`                  |
| `--plot`              | Exibir gráficos ao final        | `False`                  |
| `--save_plot CAMINHO` | Salvar gráfico 2D               | `plots/aco_meloeiro.png` |

#### Exemplo:

```bash
pdm run agro --iterations 300 --n_ants 80 --plot --verbose
```