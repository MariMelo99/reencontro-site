# Alterações da organização final

Foram removidos apenas o import e o middleware CORS duplicados em api.py. Foram preservados o adaptador ativo 03, o limiar 0,20, as rotas e a inferência. O requirements.txt foi convertido de UTF-16 para UTF-8 e recebeu o índice adicional de wheels CPU do PyTorch. A lista original foi mantida em docs/requirements_ambiente_original.txt em texto legível.

Foram acrescentados README, notebooks de análise de evidências, metadados de fontes, revisão assistida, relatório técnico e instruções de reprodução. Os JSONs congelados originais, pesos LoRA e base consultável permaneceram byte a byte iguais ao pacote recebido.

Os notebooks executam leitura e análise dos logs reais fornecidos. Nenhum treino ou inferência foi reexecutado na máquina de organização, que não tem os pesos base nem o mesmo ambiente de bibliotecas. Não há saídas simuladas apresentadas como execução de modelo.

Os hashes registrados nas avaliações correspondem ao momento em que elas foram executadas. Alterações posteriores no adaptador ativo de assistente_validado.py podem produzir diferença no hash do código atual. Isso não altera as saídas congeladas. Novas execuções devem gerar novos resultados com hashes próprios.
