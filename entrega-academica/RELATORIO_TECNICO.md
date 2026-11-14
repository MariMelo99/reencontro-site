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

## Performance experimental verificada

As células de código dos três notebooks foram reexecutadas com os arquivos originais: recuperação, análise dos logs de treinamento e análise do placar. Todas concluíram sem erro. As saídas originais dos notebooks foram preservadas; a reexecução está registrada em arquivos separados.

| Configuração | Conteúdo correto/completo | Aprovação formal | Critério estrito | Conteúdo sem suporte | Fonte/citação inválida | Tempo médio local |
|---|---:|---:|---:|---:|---:|---:|
| Sem contexto | 5/12 | 0/12 | 0/12 | 4/12 | 7/12 | 8,70 s |
| RAG simples | 9/12 | 3/12 | 3/12 | 1/12 | 2/12 | 9,45 s |
| LoRA + RAG | 10/12 | 11/12 | 9/12 | 1/12 | 1/12 | 9,71 s |

Todas as 36 saídas são JSON válido, o que não implica correção de conteúdo. Critério estrito exige conteúdo correto/completo e validação formal aprovada. As categorias de falhas podem se sobrepor. Os totais foram recalculados a partir das 36 classificações e conferidos contra o resumo.

A recuperação obteve Recall@1 e Recall@3 de 5/6 (83,33%) nas perguntas respondíveis. A calibração em 14 perguntas de desenvolvimento teve resultado perfeito, que não se manteve nas reformulações do teste.

## Demonstração de falhas reais

### j003 — fabricação no baseline e falha de recuperação
Pergunta: “A base do protótipo contém quantos registros e eles são fictícios?”

O baseline sem contexto afirmou que a base contém **0 registros**, em desacordo com os 100 registros documentados, e citou fonte não recuperada. O validador rejeitou a fonte. Na configuração LoRA + RAG, o modelo recusou responder com contexto vazio; a recusa passou formalmente, mas o sistema falhou porque a base contém a resposta. Isso distingue fabricação, recuperação e recusa indevida.

### j004 — conteúdo sem suporte aceito formalmente
A resposta LoRA + RAG acrescentou a recomendação de revisar filtros “para garantir precisão”. A revisão classificou esse complemento como não sustentado pela documentação, embora o núcleo da resposta estivesse correto. O validador formal aprovou a saída. Trata-se de um exemplo de por que citação literal não garante fundamentação de todas as afirmações.

### j006 — resposta correta rejeitada pela citação
O texto da resposta estava correto, mas a citação omitiu “fila de” e falhou na verificação literal. Esse exemplo mostra o custo do controle estrito na aceitação de respostas.

Os exemplos estão no notebook 03 e nos JSONs congelados; não são exemplos inventados ou novas execuções.

## Treinamento e seleção
Experimento 01: 16 exemplos de treino, melhor época 2, perda de validação 0,031521. Experimento 02: 28 exemplos, melhor época 3, perda 0,002076. Experimento 03: 32 exemplos e oito de validação, melhor época 2, perda 0,009320; tempo registrado de 453,3 segundos. Os experimentos diferem em dados e formato, impedindo atribuir o resultado apenas à taxa de aprendizado ou comparar perdas como se medissem a mesma tarefa.

## Limitações metodológicas
O conjunto tem 12 perguntas por configuração. A revisão é assistida por IA e exige conferência da autora; as categorias foram definidas após inspeção. Não houve avaliador humano independente. Os fatos do domínio também aparecem no treino; novas formulações não constituem teste independente amplo. O baseline sem contexto recebe instrução de usar documentos mesmo sem recebê-los. O avaliador gera em contexto vazio para registrar comportamento, enquanto a API atual recusa antes da geração. Logo, esses números não são uma medição ponta a ponta do site publicado nem uma garantia de desempenho real.

Os notebooks reanalisam evidências fornecidas; não repetem treinamento ou inferência. Os scripts de geração dos logs e avaliações e a revisão fixa do modelo base ainda precisam ser incorporados para reprodução integral. Versões do ambiente implantado não foram congeladas. O serviço gratuito tem filas e cotas.
