# Legacy Sources

Esta pasta preserva snapshots legados de exercicios que nao eram baseados em notebook.

- `ce2/`: script original e pacote `cec2017` incorporado no repositório legado.
- `ce3/`: script e layout originais do `CE3`.
- `ce3-1/`: variante legada quase duplicada de `CE3`, preservada separadamente.

Os arquivos desta pasta sao referencia historica. O fluxo canonico do repositório continua sendo `src/ce/`, `data/raw/` e a CLI `uv run ce ...`.

## Problemas Gerais Dos Legados Arquivados

- Nao sao pacotes Python modernos e nao oferecem CLI reproduzivel.
- Dependem de scripts monoliticos com estado global e forte acoplamento entre parse, algoritmo, avaliacao e exibicao.
- Nao possuem estrategia de testes automatizados nem validacoes baratas por invariantes.
- Nao possuem configuracao declarativa de seeds, parametros ou caminhos.
- Misturam codigo autoral com artefatos e layouts legados sem contratos claros.
- Devem ser tratados apenas como referencia historica; nao sao a fonte canonica de execucao.

## Resumo Por Exercicio

`ce2/`:
- Problemas principais: uso inadequado de algoritmo multiobjetivo, estado global mutavel, agregacao fragil de resultados e reshape incorreto em dimensao 10.

`ce3/`:
- Problemas principais: script monolitico, custo TSP/ATSP modelado de forma fragil, parametros hard-coded e ausencia de validacao de permutacao.

`ce3-1/`:
- Repete quase todos os problemas de `ce3/` e ainda adiciona bug no filtro de arquivos e plot forcado.
