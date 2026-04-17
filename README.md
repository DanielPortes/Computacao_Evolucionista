# Computacao Evolucionista

Repositorio consolidado para os exercicios `CE2`, `CE3`, `CE3-1`, `CE4` e `CE5`.

## Modulos Canonicos

- `ce.ex02_cec2017`: benchmark continuo CEC2017 com `GA`, `ES`, `EP`, `DE`, `PSO` e `ABC`.
- `ce.ex03_tsp`: TSP/ATSP com parse `tsplib95`, custo canonico e comparacao `GA` versus `ACO`.
- `ce.ex04_forecasting`: baseline temporal reproduzivel com `LSTM` e baseline de `GP` para o alvo `GLOBAL`.
- `ce.ex05_hpo`: busca evolucionaria barata sobre o baseline temporal com `TimeSeriesSplit`.

## Estrutura

- `src/ce/`: pacote principal
- `tests/`: testes automatizados
- `data/`: area canonica para datasets e artefatos derivados
- `legacy/`: snapshots legados preservados de `CE2`, `CE3` e `CE3-1`
- `notebooks/`: notebooks arquivados ou exploratorios
- `docs/legacy/`: documentacao da base legada

## Setup

```bash
uv sync
uv run ce --help
```

## Como Rodar

`CE2`:

```bash
uv run ce ex02 --algorithm ga --algorithm de --function-id 3 --dimension 10 --budget-multiplier 20 --n-runs 2
```

`CE3`:

```bash
uv run ce ex03 --algorithm ga --algorithm aco --instance berlin52 --generations 20 --population-size 40
```

`CE4`:

```bash
uv run ce ex04 --station rola_moca --max-epochs 5 --max-rows 240
```

`CE5`:

```bash
uv run ce ex05 --population-size 4 --generations 1 --n-splits 2 --max-epochs 2 --max-rows 160
```

## Qualidade

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

## Notebooks

Os notebooks legados foram arquivados em `notebooks/legacy/`. Eles permanecem disponiveis como referencia historica, mas nao sao a interface canonica de execucao.

O notebook principal de execucao e analise desta fase fica em `notebooks/main_execution.ipynb`. Ele orquestra os modulos canonicos via `ce.analysis.run_all` e gera figuras com `plotly`.

## Legado Arquivado

Os fontes legados preservados de `CE2`, `CE3` e `CE3-1` foram copiados para `legacy/`. Essa pasta existe para rastreabilidade historica e nao participa do fluxo canonico de lint, testes ou execucao.
