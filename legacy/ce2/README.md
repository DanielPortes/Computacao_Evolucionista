# CE2 Legacy

Snapshot do material original do repositório `CE2`.

- `mm.py`: script principal legado.
- `cec2017/`: copia do pacote incorporado no repositório antigo, incluindo `data.pkl`.

Uso canonico atual: `src/ce/ex02_cec2017/`.

## Problemas Observados

- `mm.py` modela um problema essencialmente mono-objetivo com `NSGAII`, o que adiciona complexidade sem justificativa tecnica clara.
- O script depende de estado global mutavel para selecionar funcao, dimensao e configuracao experimental.
- `test_fun()` usa esse estado global e ainda contem um loop morto que nao altera o resultado.
- A coleta dos resultados depende de indexacao fragil sobre a estrutura retornada por `experiment()`.
- A linha `reshape(-1, 2)` corrompe execucoes em dimensao `10`, porque assume incorretamente uma representacao bidimensional fixa.
- As estatisticas finais nao sao agregadas diretamente a partir do melhor valor por execucao; o fluxo reavalia estruturas intermediarias de forma indireta.
- Nao ha semente fixa declarada, o que inviabiliza reproducibilidade minima.
- Nao ha CLI, testes, contratos de configuracao nem separacao entre benchmark, algoritmo e relatorio.
- O pacote `cec2017/` foi incorporado no legado sem documentacao clara de versao ou proveniencia.
- O repositório legado original continha artefatos indevidos, como `__pycache__`, misturados ao codigo.

## Consequencia Pratica

- O material desta pasta serve apenas para rastreabilidade historica. O fluxo canonicamente corrigido e testavel esta em `src/ce/ex02_cec2017/`.
