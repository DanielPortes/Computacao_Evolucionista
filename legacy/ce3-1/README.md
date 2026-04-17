# CE3-1 Legacy

Snapshot da variante legada `CE3-1`.

- `main2.py`: script legado da variante.
- `input/`: layout original das instancias dessa variante.

Esta pasta existe para preservar a diferenca historica em relacao a `CE3`. O uso canonico atual continua sendo `src/ce/ex03_tsp/`.

## Problemas Observados

- Repete praticamente todos os problemas estruturais de `CE3`: script monolitico, globais fragis, custo TSP/ATSP mal encapsulado, parametros hard-coded, ausencia de validacao de permutacao e ausencia de testes.
- O filtro de arquivos em `main2.py` esta errado:
  `filename.endswith('.tsp') or not filename.endswith('.atsp')`
- Esse filtro praticamente desabiliza a intencao original de aceitar apenas arquivos `.tsp` e `.atsp`.
- A variante passa a forcar `plot_tour(...)`, o que adiciona efeito colateral grafico ao fluxo principal.
- Como o plot legado nao foi desenhado com `ATSP` em mente, essa decisao torna a variante ainda mais fragil.
- A variante nao resolve os problemas do `CE3`; ela apenas carrega o mesmo acoplamento com um bug adicional.

## Diferenca Historica Relevante

- `CE3` foi preservado como referencia historica principal.
- `CE3-1` foi mantido separado apenas para registrar o bug extra de filtro e a mudanca de comportamento no plot.
