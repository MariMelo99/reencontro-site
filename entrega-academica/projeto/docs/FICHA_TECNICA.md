# Ficha técnica do componente de IA

| Campo | Descrição |
| --- | --- |
| Produto | Reencontro |
| Integrantes e RM | Preencher antes da entrega |
| Cidadão e território | Pessoas que buscam orientação sobre desaparecimentos e sobre o protótipo em São Paulo |
| Problema | Dificuldade de obter orientação compreensível e fundamentada a partir de informação limitada |
| Tarefa da IA | Responder com base em documentos, apresentar evidências e reconhecer ausência de informação |
| Entrada | Pergunta de 3 a 600 caracteres, usando apenas informações fictícias na demonstração |
| Saída | JSON com status, resposta e evidências; a API pode retornar bloqueio |
| Conhecimento | Sete trechos textuais, com fontes e IDs; separado dos 100 casos fictícios do portal |
| Modelo | Qwen/Qwen3-0.6B local, com adaptador LoRA 03 |
| Recuperação | TF-IDF, similaridade de cosseno, até três trechos e limiar 0,20 |
| Integração | API FastAPI na porta 8000 e portal local na porta 5500 |
| Explicação | Fonte e citação literal exibidas no portal |
| Limites | Não confirma identidade ou localização e não encaminha relatos reais |
| Avaliação | Três configurações, 12 perguntas na avaliação mais recente, saídas brutas preservadas |
| ODS relacionado | ODS 10, inclusão por acesso à orientação; impacto social ainda não medido em campo |
| URL do produto e repositório | Preencher após publicação; a demonstração atual é local |
