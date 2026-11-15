import json
from pathlib import Path

PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_DADOS = PASTA_PROJETO / "dados"
PASTA_DADOS.mkdir(exist_ok=True)

# Cenários fictícios usados exclusivamente no experimento acadêmico.
# As paráfrases de cada cenário permanecem na mesma divisão.
cenarios = [
    (
        "treino", "t001",
        "Não é necessário comprar ingresso para acessar a exposição.",
        [
            "Preciso comprar ingresso para acessar a exposição?",
            "É obrigatório comprar ingresso para entrar na exposição?",
        ],
        "Não é necessário comprar ingresso para acessar a exposição.",
    ),
    (
        "treino", "t002",
        "É necessário apresentar o comprovante no atendimento.",
        [
            "Preciso apresentar o comprovante?",
            "O comprovante é necessário no atendimento?",
        ],
        "É necessário apresentar o comprovante no atendimento.",
    ),
    (
        "treino", "t003",
        "A unidade não atende aos domingos.",
        [
            "A unidade atende aos domingos?",
            "Posso receber atendimento nessa unidade no domingo?",
        ],
        "A unidade não atende aos domingos.",
    ),
    (
        "treino", "t004",
        "O posto funciona às segundas e quartas-feiras.",
        [
            "Em quais dias o posto funciona?",
            "Quais são os dias de funcionamento do posto?",
        ],
        "O posto funciona às segundas e quartas-feiras.",
    ),
    (
        "treino", "t005",
        "O balcão oferece orientação sobre o formulário.",
        [
            "Qual é o telefone do balcão?",
            "Qual número posso ligar para falar com o balcão?",
        ],
        None,
    ),
    (
        "treino", "t006",
        "O envio do formulário não garante aprovação do pedido.",
        [
            "Enviar o formulário garante aprovação?",
            "Meu pedido será aprovado só porque enviei o formulário?",
        ],
        "O envio do formulário não garante aprovação do pedido.",
    ),
    (
        "treino", "t007",
        "Não é permitido fotografar os documentos no balcão.",
        [
            "Posso fotografar os documentos no balcão?",
            "É permitido tirar fotos dos documentos no balcão?",
        ],
        "Não é permitido fotografar os documentos no balcão.",
    ),
    (
        "treino", "t008",
        "A análise das solicitações é realizada por uma pessoa da equipe.",
        [
            "Quem analisa as solicitações?",
            "A análise é realizada por quem?",
        ],
        "A análise das solicitações é realizada por uma pessoa da equipe.",
    ),
    (
        "treino", "t009",
        "O espaço oferece aulas de música.",
        [
            "Quanto custa a aula de música?",
            "Qual é o valor da aula de música?",
        ],
        None,
    ),
    (
        "treino", "t010",
        "A unidade realiza atendimento presencial.",
        [
            "Qual é o endereço da unidade?",
            "Em qual rua fica a unidade?",
        ],
        None,
    ),
    (
        "treino", "t011",
        "As solicitações passam por análise da equipe.",
        [
            "Em quantos dias minha solicitação será analisada?",
            "Qual é o prazo para concluir a análise?",
        ],
        None,
    ),
    (
        "treino", "t012",
        "O formulário foi recebido e aguarda avaliação.",
        [
            "Meu pedido foi aprovado?",
            "Já posso considerar meu pedido aprovado?",
        ],
        None,
    ),
    (
        "treino", "t013",
        "O relatório informa que o objeto foi perdido na segunda-feira.",
        [
            "Onde está o objeto agora?",
            "Qual é a localização atual do objeto?",
        ],
        None,
    ),
    (
        "treino", "t014",
        "A oficina acontece aos sábados.",
        [
            "Qual é o nome de quem ministra a oficina?",
            "Quem é o responsável por ministrar a oficina?",
        ],
        None,
    ),
    (
        "validacao", "v001",
        "Não é necessário agendar a visita ao espaço cultural.",
        [
            "Preciso agendar a visita ao espaço cultural?",
            "O agendamento é obrigatório para visitar o espaço cultural?",
        ],
        "Não é necessário agendar a visita ao espaço cultural.",
    ),
    (
        "validacao", "v002",
        "O centro oferece atividades de leitura.",
        [
            "Qual é o endereço do centro?",
            "Em qual rua fica o centro?",
        ],
        None,
    ),
    (
        "teste", "e001",
        "Não é necessário imprimir o recibo para retirar o material.",
        [
            "Preciso imprimir o recibo para retirar o material?",
            "É obrigatório imprimir o recibo para retirar o material?",
        ],
        "Não é necessário imprimir o recibo para retirar o material.",
    ),
    (
        "teste", "e002",
        "O núcleo oferece oficinas de pintura.",
        [
            "Quanto custa a oficina de pintura?",
            "Qual é o valor cobrado pela oficina de pintura?",
        ],
        None,
    ),
]

divisoes = {
    "treino": [],
    "validacao": [],
    "teste": [],
}

ids_cenarios = set()

for divisao, id_cenario, contexto, perguntas, resposta in cenarios:
    if id_cenario in ids_cenarios:
        raise ValueError(f"ID de cenário duplicado: {id_cenario}")

    ids_cenarios.add(id_cenario)

    for numero, pergunta in enumerate(perguntas, start=1):
        tem_resposta = resposta is not None

        if tem_resposta:
            resposta_esperada = f"{resposta} [{id_cenario}]"
            fontes_esperadas = [id_cenario]
        else:
            resposta_esperada = (
                "Essa informação não consta na base disponível."
            )
            fontes_esperadas = []

        exemplo = {
            "id": f"{id_cenario}_{numero}",
            "cenario": id_cenario,
            "origem": "Cenário fictício autoral para experimento",
            "contexto": f"[{id_cenario}] {contexto}",
            "pergunta": pergunta,
            "resposta": resposta_esperada,
            "tem_resposta": tem_resposta,
            "fontes_esperadas": fontes_esperadas,
        }

        divisoes[divisao].append(exemplo)

contagens = {}

for divisao, exemplos in divisoes.items():
    caminho = PASTA_DADOS / f"{divisao}.jsonl"

    with caminho.open("w", encoding="utf-8") as arquivo:
        for exemplo in exemplos:
            arquivo.write(
                json.dumps(exemplo, ensure_ascii=False) + "\n"
            )

    com_resposta = sum(
        exemplo["tem_resposta"] for exemplo in exemplos
    )
    sem_resposta = len(exemplos) - com_resposta

    contagens[divisao] = {
        "total": len(exemplos),
        "com_resposta": com_resposta,
        "sem_resposta": sem_resposta,
    }

    print(
        f"{divisao}: {len(exemplos)} exemplos "
        f"({com_resposta} com resposta e {sem_resposta} sem resposta)"
    )
    print(f"Arquivo: {caminho}\n")

manifesto = {
    "versao": "experimento_02",
    "finalidade": "Experimento com treino equilibrado",
    "conteudo": "Cenários fictícios, sem dados pessoais",
    "separacao": "Por cenário, mantendo paráfrases na mesma divisão",
    "quantidades": contagens,
    "observacao": (
        "O conjunto teste.jsonl já foi analisado no experimento 01. "
        "Seu uso posterior é diagnóstico. A avaliação final requer "
        "novos cenários reservados."
    ),
}

(PASTA_DADOS / "manifesto.json").write_text(
    json.dumps(manifesto, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Arquivos preparados.")