# Segurança e privacidade do protótipo

O uso demonstrado é acadêmico e local. Os exemplos de treinamento são fictícios e a interface orienta a não enviar dados pessoais. O servidor é iniciado em 127.0.0.1. CORS aceita apenas duas origens locais na porta 5500, mas não substitui autenticação.

A API limita a pergunta a 3 a 600 caracteres, impede gerações simultâneas com uma trava, valida a estrutura JSON, exige IDs recuperados e citações literais e devolve uma mensagem controlada quando a validação falha. A busca vazia produz recusa antes da geração. A interface apresenta fonte e trecho e não usa HTML gerado pelo modelo para renderizar o conteúdo.

A pergunta, a resposta e os documentos passam pela memória do processo local. A API não implementa persistência do corpo das consultas. O servidor pode registrar metadados HTTP, como rota e endereço. A geração local não elimina riscos de acesso ao computador ou de coleta indevida no campo de texto. A interface não detecta nem anonimiza automaticamente dados pessoais.

A validação não prova pertinência nem equivalência semântica. Uma resposta pode citar um trecho verdadeiro e conter extrapolação. Os testes de injeção documentaram bloqueio de citação alterada e cópia do placeholder, sem demonstrar resistência geral a ataques.

O botão Limpar consulta remove o conteúdo exibido no painel, não logs do sistema ou dados de outras partes do site. A versão não implementa contas, autenticação, criptografia de disco, política automatizada de retenção, gestão de direitos do titular nem operação de um cadastro real de desaparecidos. Esses pontos precisam ser definidos e avaliados antes de qualquer uso com dados reais, especialmente de crianças e adolescentes.

A referência de orientação consultada é o guia de segurança da informação da ANPD. Esta documentação descreve controles técnicos e pendências, sem certificação de conformidade integral à LGPD.

https://www.gov.br/anpd/pt-br/centrais-de-conteudo/materiais-educativos-e-publicacoes/guia-vf.pdf
