from pathlib import Path

import avaliacao_final as avaliacao


PASTA = Path(__file__).resolve().parent

avaliacao.ADAPTADOR = (
    PASTA / "artefatos" / "lora_experimento_03"
)

# Novas formulações sobre os documentos existentes.
# O domínio e os fatos já apareceram no treino.
# A avaliação mede generalização de perguntas e uso do contexto.
avaliacao.CASOS = [
    {
        "id": "j001",
        "pergunta": (
            "Um familiar desapareceu há poucas horas. "
            "Devo aguardar até amanhã para registrar o boletim?"
        ),
        "fontes": ["oficial_001"],
        "gabarito": (
            "Não é necessário esperar 24 horas para comunicar "
            "o desaparecimento e registrar o boletim de ocorrência."
        ),
    },
    {
        "id": "j002",
        "pergunta": (
            "Já fiz a comunicação pelo SINAL DESAPARECIDOS. "
            "Isso elimina a necessidade de registrar ocorrência?"
        ),
        "fontes": ["oficial_002"],
        "gabarito": (
            "A comunicação pelo SINAL DESAPARECIDOS não dispensa "
            "o registro de ocorrência na polícia judiciária."
        ),
    },
    {
        "id": "j003",
        "pergunta": (
            "A base do protótipo contém quantos registros "
            "e eles são fictícios?"
        ),
        "fontes": ["produto_001"],
        "gabarito": (
            "O protótipo contém 100 registros fictícios."
        ),
    },
    {
        "id": "j004",
        "pergunta": (
            "Sem conhecer o nome, posso iniciar a busca "
            "por características e depois usar filtros?"
        ),
        "fontes": ["produto_002"],
        "gabarito": (
            "É possível iniciar a consulta por características "
            "e refinar os resultados com os filtros disponíveis."
        ),
    },
    {
        "id": "j005",
        "pergunta": (
            "A comparação visual da busca por foto "
            "é suficiente para confirmar a identidade?"
        ),
        "fontes": ["produto_003"],
        "gabarito": (
            "A comparação visual não confirma identidade. "
            "Uma possível correspondência exige revisão humana."
        ),
    },
    {
        "id": "j006",
        "pergunta": (
            "Para encaminhar uma informação adicional "
            "à revisão humana, devo abrir a ficha?"
        ),
        "fontes": ["produto_004"],
        "gabarito": (
            "O usuário pode abrir a ficha e escrever uma informação "
            "adicional para a fila de revisão humana. "
            "O fluxo do protótipo é simulado."
        ),
    },
    {
        "id": "j007",
        "pergunta": (
            "Qual é o número de telefone do suporte "
            "para uma pessoa com dificuldade no portal?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "j008",
        "pergunta": (
            "Qual é o nome do aeroporto que já recebeu "
            "um totem do Reencontro?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "j009",
        "pergunta": (
            "Qual é a quantidade de atendentes que revisam "
            "os relatos do Reencontro?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "j010",
        "pergunta": (
            "Qual é o custo mensal de hospedagem do portal?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "j011",
        "pergunta": (
            "Qual é a data prevista para lançar "
            "o aplicativo na loja de celulares?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "j012",
        "pergunta": (
            "Em qual bairro funciona a sede administrativa "
            "do Reencontro?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
]


if __name__ == "__main__":
    avaliacao.main()