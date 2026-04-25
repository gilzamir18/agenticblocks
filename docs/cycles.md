# Nested Cycles: Implementação Concluída

A funcionalidade de ciclos aninhados foi totalmente incorporada ao `agenticblocks`. Agora, desenvolvedores podem encapsular uma lógica de validação dentro de outra de forma modular.

## O que foi feito

### 1. Grafo Hierárquico (`graph.py`)
- Foram introduzidas as funções auxiliares `_get_actual_entry` e `_get_actual_condition`. Quando você conecta a algo chamado `intention_loop`, o grafo interno sabe descer na árvore de aninhamento até o bloco físico (como `get_user_input` ou `check_intention`).
- A função `collapsed_graph()` agora agrupa apenas os "top-level cycles", resolvendo o grafo do nível macro (usado para gerar *waves* e validar ciclos acidentais).

### 2. Executor Recursivo (`executor.py`)
- O método de coleção de dependências `_collect_inputs` e `_collect_cycle_entry_inputs` foi adaptado com um seletor de "top_cycle". Isso previne que blocos roubem outputs do meio da execução de ciclos alheios, garantindo o escopo das dependências.
- O bloco central de processamento em `_execute_cycle` agora identifica instâncias de nós virtuais. Se o próximo item do `chain` for um sub-ciclo, o executor se chama recursivamente enviando o input mapeado (`override_input`), aguardando a finalização total daquele loop antes de seguir para o próximo bloco.

### 3. Exemplo de Validação (`examples/06_nested_cycles.py`)
Criamos um fluxo de "vendedor x checador de intenção" e verificamos com sucesso sua execução:
- O ciclo interno roda primeiro até a intenção ser validada.
- O fluxo segue para o "vendedor", que identifica um dado faltando (ex: sem endereço).
- O validador externo (`check_done`) falha e injeta um feedback "Please provide your address".
- O ciclo externo engatilha de volta, relançando **todo** o fluxo interno com o input atualizado com o histórico de rejeições.

## Como Utilizar

Você pode definir normalmente os ciclos declarando-os na `sequence` de outros ciclos:

```python
    graph.add_cycle(
        name="intention_loop",
        sequence=["get_user_input", "intention_agent", "check_intention"],
        condition_block="check_intention",
        max_iterations=3,
    )

    # O "intention_loop" entra nativamente como membro
    graph.add_cycle(
        name="refine_loop",
        sequence=["intention_loop", "sales_agent", "check_done"],
        condition_block="check_done",
        max_iterations=2,
    )
```

> [!TIP]
> Use essa técnica para modularizar comportamentos. Se um agente precisa lidar sistematicamente com ferramentas antes de escrever um artigo, isole a etapa de "pesquisa" em um ciclo aninhado com sua própria condição.

> [!WARNING]
> Como previsto, cada rejeição de nível superior aciona completamente os ciclos de nível inferior novamente. O uso do LLM crescerá exponencialmente dependendo do valor ajustado em `max_iterations`. Monitore isso em produção.
