# Integração com o portal existente

Copie index.html para a pasta original do site, que já contém styles.css, app.js, foto.js, foto-ui.js, data/casos.js e assets/. Faça cópia do HTML anterior antes da substituição.

O arquivo usa http://127.0.0.1:8000/consultar. Sirva a pasta completa do site na porta 5500, com o comando abaixo. Mantenha também a API na porta 8000.

```powershell
py -3.12 -m http.server 5500 --bind 127.0.0.1
```

Abra http://127.0.0.1:5500. O HTML é o complemento de integração. Os estilos, scripts originais, fotos e 100 casos do portal não foram incluídos no ZIP enviado para organização. Eles devem ser adicionados ao repositório do produto ou mantidos na pasta original para a demonstração completa. O HTML isolado não reproduz todas as telas.
