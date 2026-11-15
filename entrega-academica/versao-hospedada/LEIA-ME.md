# Versão hospedada

O arquivo app.py corresponde ao ajuste do Space Gradio/ZeroGPU, com carregamento do adaptador na CPU e normalização da pontuação final. O servidor publicado usa MariAstro4/reencontro-assistente.

Esta pasta documenta as diferenças da hospedagem e não é um projeto independente: para montar o Space, envie app.py e requirements.txt junto com assistente_validado.py, busca.py, base/ e artefatos/lora_experimento_03/ da pasta projeto/. Use SDK Gradio e ZeroGPU. O requirements desta pasta complementa as dependências gerenciadas pelo Space; não é um freeze do ambiente local.

O site na raiz do GitHub já consulta o Space. A integração_site/index.html da pasta projeto preserva a versão local original, com API em localhost, e não deve substituir o HTML publicado.
