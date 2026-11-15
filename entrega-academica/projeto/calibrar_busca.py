import json
from pathlib import Path

from busca import BuscaConhecimento


PASTA = Path(__file__).resolve().parent
SAIDA = PASTA / "avaliacao" / "calibracao_busca.json"

# Conjunto de desenvolvimento usado para escolher o limiar.
# Não representa uma avaliação final independente.
CASOS = [
    (
        "Preciso esperar 24 horas para registrar o desaparecimento?",
        ["oficial_001"],
    ),
    (
        "O SINAL DESAPARECIDOS substitui o registro de ocorrência?",
        ["oficial_002"],
    ),
    (
        "Qual é a finalidade do projeto Reencontro?",
        ["produto_001"],
    ),
    (
        "Posso consultar casos pelo nome ou por características?",
        ["produto_002"],
    ),
    (
        "Uma foto semelhante confirma a identidade?",
        ["produto_003"],
    ),
    (
        "Como enviar informações adicionais para revisão?",
        ["produto_004"],
    ),
    (
        "A base contém localização atual verificada?",
        ["produto_005"],
    ),
    (
        "Qual é o telefone da equipe do Reencontro?",
        [],
    ),
    (
        "Qual é o endereço do primeiro totem?",
        [],
    ),
    (
        "Quanto custa instalar um totem?",
        [],
    ),
    (
        "Em quantas horas a equipe revisa um relato?",
        [],
    ),
    (
        "Quem é o responsável técnico pelo projeto?",
        [],
    ),
    (
        "Qual é o horário de atendimento presencial?",
        [],
    ),
    (
        "Qual é o e-mail para falar com a equipe?",
        [],
    ),
]


def avaliar_limiar(registros, limiar):
    verdadeiros_positivos = 0
    falsos_negativos = 0
    verdadeiros_negativos = 0
    falsos_positivos = 0

    for registro in registros:
        tem_resposta = bool(registro["fontes_esperadas"])
        aceita = (
            bool(registro["recuperados"])
            and registro["pontuacao_maxima"] >= limiar
        )

        if tem_resposta:
            if aceita:
                verdadeiros_positivos += 1
            else:
                falsos_negativos += 1
        else:
            if aceita:
                falsos_positivos += 1
            else:
                verdadeiros_negativos += 1

    sensibilidade = verdadeiros_positivos / (
        verdadeiros_positivos + falsos_negativos
    )
    especificidade = verdadeiros_negativos / (
        verdadeiros_negativos + falsos_positivos
    )

    return {
        "limiar": limiar,
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
        "acuracia_balanceada": (
            sensibilidade + especificidade
        ) / 2,
        "respondiveis_aceitas": verdadeiros_positivos,
        "respondiveis_rejeitadas": falsos_negativos,
        "sem_resposta_aceitas": falsos_positivos,
        "sem_resposta_rejeitadas": verdadeiros_negativos,
    }


def calcular_recall(registros, quantidade, limiar):
    valores = []

    for registro in registros:
        esperadas = set(registro["fontes_esperadas"])

        if not esperadas:
            continue

        recuperados = [
            item["id"]
            for item in registro["recuperados"]
            if item["pontuacao"] >= limiar
        ][:quantidade]

        valores.append(
            len(esperadas.intersection(recuperados)) / len(esperadas)
        )

    return sum(valores) / len(valores)


def main():
    if SAIDA.exists():
        raise FileExistsError(
            f"O resultado já existe em {SAIDA}. "
            "Altere o nome de SAIDA para preservar a execução anterior."
        )

    busca = BuscaConhecimento()
    registros = []

    for pergunta, fontes_esperadas in CASOS:
        recuperados = busca.buscar(pergunta, quantidade=3)

        pontuacao_maxima = max(
            (item["pontuacao"] for item in recuperados),
            default=0.0,
        )

        registros.append(
            {
                "pergunta": pergunta,
                "fontes_esperadas": fontes_esperadas,
                "pontuacao_maxima": pontuacao_maxima,
                "recuperados": recuperados,
            }
        )

    # Compara limiares de 0,00 a 1,00, em intervalos de 0,02.
    candidatos = [
        avaliar_limiar(registros, numero / 100)
        for numero in range(0, 101, 2)
    ]

    # Prioriza acurácia balanceada.
    # Em empate, preserva mais perguntas respondíveis.
    # Persistindo o empate, escolhe o menor limiar.
    melhor = max(
        candidatos,
        key=lambda item: (
            item["acuracia_balanceada"],
            item["sensibilidade"],
            -item["limiar"],
        ),
    )

    limiar = melhor["limiar"]

    metricas = {
        "recall_1_antes": calcular_recall(registros, 1, 0.0),
        "recall_3_antes": calcular_recall(registros, 3, 0.0),
        "recall_1_depois": calcular_recall(registros, 1, limiar),
        "recall_3_depois": calcular_recall(registros, 3, limiar),
    }

    relatorio = {
        "tipo": "Calibração em conjunto de desenvolvimento",
        "criterio": (
            "Maior acurácia balanceada, com desempate por maior "
            "sensibilidade e depois menor limiar"
        ),
        "melhor_configuracao": melhor,
        "metricas_recuperacao": metricas,
        "candidatos": candidatos,
        "casos": registros,
    }

    SAIDA.parent.mkdir(exist_ok=True)

    SAIDA.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Limiar candidato: {limiar:.2f}")
    print(
        "Acurácia balanceada: "
        f"{melhor['acuracia_balanceada']:.1%}"
    )
    print(
        "Perguntas respondíveis aceitas: "
        f"{melhor['respondiveis_aceitas']}/7"
    )
    print(
        "Perguntas sem resposta aceitas: "
        f"{melhor['sem_resposta_aceitas']}/7"
    )
    print(
        "Perguntas sem resposta rejeitadas: "
        f"{melhor['sem_resposta_rejeitadas']}/7"
    )
    print(f"Recall@1 antes: {metricas['recall_1_antes']:.1%}")
    print(f"Recall@3 antes: {metricas['recall_3_antes']:.1%}")
    print(f"Recall@1 depois: {metricas['recall_1_depois']:.1%}")
    print(f"Recall@3 depois: {metricas['recall_3_depois']:.1%}")
    print(f"\nResultado salvo em: {SAIDA}")


if __name__ == "__main__":
    main()