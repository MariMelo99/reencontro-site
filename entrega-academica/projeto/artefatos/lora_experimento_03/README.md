---
base_model: Qwen/Qwen3-0.6B
library_name: peft
language:
- pt
---
# Adaptador local Reencontro experimento 03

LoRA com rank 8, alpha 16, dropout 0,05 em q_proj e v_proj. Treinamento pela CPU em float32. Foram 32 exemplos de treino em JSON e 8 de validação. Melhor época 2, perda de validação 0,009320. A pasta contém o adaptador selecionado e os arquivos de dados efetivamente usados.

O uso é demonstrativo, com contexto documental e revisão humana. O modelo base está identificado por nome, sem commit original fixado. Ver treinamento.json, treino_json.json e validacao_json.json. Licença do modelo base conforme seu model card, Apache 2.0. A licença de distribuição do trabalho autoral deve ser definida pela autora, respeitando os componentes utilizados.

Os resultados completos e as limitações estão no README da raiz e em avaliacao/. Não aplicar a decisões sobre pessoas reais ou confirmação de identidade.
