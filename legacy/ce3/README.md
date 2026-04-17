# CE3 Legacy

Snapshot do material original do repositório `CE3`.

- `main2.py`: script monolitico legado.
- `input/`: layout original das instancias usadas pelo script.

Uso canonico atual: `src/ce/ex03_tsp/`.

## Problemas Observados

- `main2.py` mistura parse de instancias, construcao de dados, avaliacao, operadores do GA, metricas, plot e impressao em um unico script.
- O script depende de globais como `problem`, `graph` e `cities`, o que torna o fluxo fragil e dificil de testar.
- A avaliacao de instancias `TSP` usa distancia euclidiana crua sobre coordenadas, em vez de usar a definicao canonica de custo da propria instancia TSPLIB.
- A avaliacao de instancias `ATSP` passa por uma conversao de indices e estrutura de grafo que pode quebrar o enderecamento original da instancia.
- O mesmo script tenta servir tanto `TSP` quanto `ATSP`, mas com caminhos de custo e representacao pouco isolados.
- Os parametros do algoritmo genetico sao hard-coded no script.
- Nao ha validacao explicita de que cada individuo continua sendo uma permutacao valida.
- Nao ha controle declarativo de seed ou reproducibilidade.
- Nao ha testes automatizados, smoke tests ou validacoes por invariantes.
- O plot e o benchmark estao acoplados ao mesmo fluxo de execucao, em vez de serem passos separados.

## Consequencia Pratica

- Esta pasta preserva o comportamento historico do script, mas o fluxo canonico corrigido e rastreavel esta em `src/ce/ex03_tsp/`.
