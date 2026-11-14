# Relatório técnico parcial — Reencontro

## Problema e solução
O protótipo demonstra consulta de casos fictícios, encaminhamento simulado para revisão e orientação documental. O assistente usa recuperação de informação e modelo Qwen3-0.6B com adaptador LoRA. A aplicação é acadêmica; não identifica pessoas nem verifica localização atual.

## Arquitetura e implementação
A interface estática roda na Vercel. As chamadas de geração são feitas ao Gradio público no Hugging Face ZeroGPU. A base documental possui sete trechos. A recuperação TF-IDF usa unigramas e bigramas, remoção de acentos, similaridade de cosseno e limiar 0,20. Até três documentos recuperados compõem o contexto do modelo.

O adaptador usa rank 8, alpha 16 e módulos q_proj/v_proj, conforme adapter_config.json. A inferência configura do_sample=False e até 350 novos tokens. Os pesos são inicialmente carregados na CPU e o modelo é movido para CUDA na configuração do ZeroGPU.

## Controles de saída
A instrução pede uso exclusivo dos documentos e resposta JSON. O validador exige status, resposta e evidências, verifica IDs e citações literais e bloqueia estruturas inválidas. Esses controles são formais: uma frase citada pode não sustentar a conclusão; uma recusa padronizada também pode ser indevida. A interface informa a necessidade de revisão humana.

## Evidências observadas na publicação
- O carrossel recuperou o funcionamento após inclusão de data/casos.js e correção dos caminhos das imagens para a raiz.
- O assistente carregou no ZeroGPU após especificar torch_device="cpu" no carregamento do adaptador.
- Houve resposta fundamentada sobre a finalidade do projeto, com evidência produto_001 e validação formal aprovada.
- Houve recusa indevida para pergunta sobre o mesmo tema, apesar da existência do trecho. Na reprodução local da busca, as versões com e sem interrogação recuperaram produto_001 com similaridade aproximada 0,2342, acima do limiar; isso não reproduz isoladamente a inferência do modelo.
- A normalização da pontuação final foi adicionada. Uma consulta direta posterior com interrogação retornou resposta aprovada. Não foi demonstrada eliminação geral do problema.
- O serviço bloqueou consultas por esgotamento da cota gratuita de GPU.

## Performance e alucinações
Não foram fornecidos nesta etapa notebooks executados de avaliação ou um conjunto de teste independente completo. Portanto, não se atribuem métricas de qualidade ou latência ao modelo. Os eventos acima são observações manuais, não uma avaliação estatística.

Uma recusa indevida é um erro de resposta, mas não constitui por si só demonstração de fabricação factual. A entrega deve acrescentar exemplos documentados de alucinação e avaliar se o validador bloqueia ou aceita cada caso, preservando as saídas originais. Não foram inventados exemplos como se fossem execuções reais.

## Limites e próximos passos
Validar estabilidade entre formulações; separar falha de recuperação de recusa do modelo; medir resultados em conjunto independente; revisar semântica das citações; documentar cotas e dependências. A hospedagem gratuita reduz custos, mas introduz filas e limites de disponibilidade.
