import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from avaliar_experimento_02 import gerar, INSTRUCAO
from busca import BuscaConhecimento


MODELO = "Qwen/Qwen3-0.6B"
PASTA = Path(__file__).resolve().parent
ADAPTADOR = PASTA / "artefatos" / "lora_experimento_02"
PASTA_RESULTADOS = PASTA / "avaliacao"
SAIDA = PASTA_RESULTADOS / "comparacao_reencontro_01.json"

# Perguntas de desenvolvimento.
# Parte dos temas já foi usada nos diagnósticos anteriores.
# Esta avaliação não deve ser apresentada como teste final independente.
CASOS = [
    {
        "id": "r001",
        "pergunta": (
            "É preciso aguardar 24 horas para comunicar "
            "o desaparecimento?"
        ),
        "fontes_esperadas": ["oficial_001"],
        "gabarito": (
            "Não é necessário esperar 24 horas para comunicar "
            "o desaparecimento e registrar o boletim de ocorrência."
        ),
    },
    {
        "id": "r002",
        "pergunta": (
            "A comunicação pelo SINAL DESAPARECIDOS dispensa "
            "o registro de ocorrência?"
        ),
        "fontes_esperadas": ["oficial_002"],
        "gabarito": (
            "A comunicação pelo SINAL DESAPARECIDOS não dispensa "
            "o registro de ocorrência na polícia judiciária."
        ),
    },
    {
        "id": "r003",
        "pergunta": (
            "Quantos registros fictícios existem no protótipo Reencontro?"
        ),
        "fontes_esperadas": ["produto_001"],
        "gabarito": (
            "O protótipo utiliza 100 registros fictícios."
        ),
    },
    {
        "id": "r004",
        "pergunta": (
            "A semelhança entre fotos confirma a identidade de uma pessoa?"
        ),
        "fontes_esperadas": ["produto_003"],
        "gabarito": (
            "A semelhança entre fotos não confirma identidade. "
            "Uma possível correspondência exige revisão humana."
        ),
    },
    {
        "id": "r005",
        "pergunta": (
            "Como enviar uma informação adicional sobre um caso?"
        ),
        "fontes_esperadas": ["produto_004"],
        "gabarito": (
            "O usuário pode abrir a ficha e escrever uma informação "
            "adicional para a fila de revisão humana. "
            "No protótipo, o fluxo é simulado com dados fictícios."
        ),
    },
    {
        "id": "r006",
        "pergunta": (
            "Qual é o endereço de instalação do primeiro totem?"
        ),
        "fontes_esperadas": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "r007",
        "pergunta": (
            "Qual é o telefone da equipe de atendimento do Reencontro?"
        ),
        "fontes_esperadas": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "r008",
        "pergunta": (
            "Em quantas horas a equipe revisa cada informação adicional?"
        ),
        "fontes_esperadas": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
]


def montar_contexto(resultados):
    return "\n\n".join(
        f"[{item['id']}]\n"
        f"Título: {item['titulo']}\n"
        f"Conteúdo: {item['texto']}"
        for item in resultados
    )


def calcular_recall(ids_recuperados, ids_esperados, quantidade):
    encontrados = set(ids_recuperados[:quantidade])
    esperados = set(ids_esperados)

    return len(encontrados & esperados) / len(esperados)


def main():
    if SAIDA.exists():
        raise FileExistsError(
            f"O resultado já existe em {SAIDA}. "
            "Altere o nome de SAIDA para preservar a execução anterior."
        )

    if not (ADAPTADOR / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Adaptador não encontrado em {ADAPTADOR}"
        )

    torch.set_num_threads(8)
    torch.manual_seed(42)

    busca = BuscaConhecimento()

    print("Carregando modelo e adaptador...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODELO,
        local_files_only=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODELO,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(
        base,
        ADAPTADOR,
        is_trainable=False,
    )
    model.config.use_cache = True
    model.eval()

    resultados = []
    valores_recall_1 = []
    valores_recall_3 = []

    for numero, caso in enumerate(CASOS, start=1):
        recuperados = busca.buscar(
            caso["pergunta"],
            quantidade=3,
        )

        ids_recuperados = [
            item["id"] for item in recuperados
        ]

        contexto = montar_contexto(recuperados)

        entrada_sem_contexto = {
            "pergunta": caso["pergunta"],
            "contexto": "Nenhum documento foi fornecido.",
        }

        entrada_com_contexto = {
            "pergunta": caso["pergunta"],
            "contexto": contexto or "Nenhum documento foi recuperado.",
        }

        with model.disable_adapter():
            sem_contexto = gerar(
                model,
                tokenizer,
                entrada_sem_contexto,
            )
            rag_simples = gerar(
                model,
                tokenizer,
                entrada_com_contexto,
            )

        lora_rag = gerar(
            model,
            tokenizer,
            entrada_com_contexto,
        )

        recall_1 = None
        recall_3 = None

        # Perguntas sem fonte de resposta ficam fora do recall.
        if caso["fontes_esperadas"]:
            recall_1 = calcular_recall(
                ids_recuperados,
                caso["fontes_esperadas"],
                1,
            )
            recall_3 = calcular_recall(
                ids_recuperados,
                caso["fontes_esperadas"],
                3,
            )
            valores_recall_1.append(recall_1)
            valores_recall_3.append(recall_3)

        registro = {
            **caso,
            "trechos_recuperados": recuperados,
            "recall_1": recall_1,
            "recall_3": recall_3,
            "a_original_sem_contexto": sem_contexto,
            "b_original_rag": rag_simples,
            "c_lora_rag": lora_rag,
        }
        resultados.append(registro)

        print(f"\nTESTE {numero}/{len(CASOS)}")
        print(f"PERGUNTA: {caso['pergunta']}")
        print(f"GABARITO: {caso['gabarito']}")
        print(f"FONTES RECUPERADAS: {ids_recuperados}")
        print(f"A — SEM CONTEXTO: {sem_contexto['resposta']}")
        print(f"B — RAG SIMPLES: {rag_simples['resposta']}")
        print(f"C — LORA + RAG: {lora_rag['resposta']}", flush=True)

    media_recall_1 = (
        sum(valores_recall_1) / len(valores_recall_1)
    )
    media_recall_3 = (
        sum(valores_recall_3) / len(valores_recall_3)
    )

    relatorio = {
        "modelo": MODELO,
        "adaptador": ADAPTADOR.name,
        "tipo_avaliacao": "Desenvolvimento",
        "instrucao_comum": INSTRUCAO,
        "geracao": {
            "do_sample": False,
            "max_new_tokens": 120,
        },
        "recuperacao": {
            "metodo": "TF-IDF e similaridade de cosseno",
            "quantidade_maxima": 3,
            "filtro_atual": "Pontuação maior que zero",
            "limiar_calibrado": False,
        },
        "recall_1": media_recall_1,
        "recall_3": media_recall_3,
        "perguntas_com_fonte_esperada": len(valores_recall_1),
        "resultados": resultados,
    }

    PASTA_RESULTADOS.mkdir(exist_ok=True)

    SAIDA.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nRecall@1: {media_recall_1:.1%}")
    print(f"Recall@3: {media_recall_3:.1%}")
    print(f"Resultado salvo em: {SAIDA}")


if __name__ == "__main__":
    main()