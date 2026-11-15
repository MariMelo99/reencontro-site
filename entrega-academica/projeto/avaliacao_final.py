import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from assistente_validado import (
    ADAPTADOR,
    INSTRUCAO,
    MODELO,
    validar_saida,
)
from busca import BuscaConhecimento


PASTA = Path(__file__).resolve().parent
LIMIAR = 0.20

# Novas formulações sobre a mesma base de conhecimento.
# A cobertura continua limitada aos documentos do protótipo.
CASOS = [
    {
        "id": "f001",
        "pergunta": (
            "O registro do boletim de ocorrência pode ser feito "
            "antes de completar um dia de desaparecimento?"
        ),
        "fontes": ["oficial_001"],
        "gabarito": (
            "Não é necessário esperar 24 horas para registrar "
            "o boletim de ocorrência."
        ),
    },
    {
        "id": "f002",
        "pergunta": (
            "Se eu comunicar o desaparecimento à PRF pelo SINAL, "
            "ainda preciso registrar ocorrência na polícia judiciária?"
        ),
        "fontes": ["oficial_002"],
        "gabarito": (
            "A comunicação pelo SINAL não dispensa o registro "
            "de ocorrência na polícia judiciária."
        ),
    },
    {
        "id": "f003",
        "pergunta": (
            "Os cem registros do Reencontro representam "
            "pessoas reais ou casos inventados?"
        ),
        "fontes": ["produto_001"],
        "gabarito": (
            "São 100 registros fictícios utilizados no protótipo acadêmico."
        ),
    },
    {
        "id": "f004",
        "pergunta": (
            "Tenho apenas algumas características da pessoa. "
            "Consigo começar uma consulta no protótipo?"
        ),
        "fontes": ["produto_002"],
        "gabarito": (
            "É possível iniciar a consulta por características "
            "e refinar os resultados com os filtros disponíveis."
        ),
    },
    {
        "id": "f005",
        "pergunta": (
            "Duas imagens parecidas já são prova de que "
            "se trata da mesma pessoa?"
        ),
        "fontes": ["produto_003"],
        "gabarito": (
            "Semelhança não confirma identidade. "
            "Uma possível correspondência exige revisão humana."
        ),
    },
    {
        "id": "f006",
        "pergunta": (
            "Onde escrevo um relato adicional para ser "
            "encaminhado à revisão humana?"
        ),
        "fontes": ["produto_004"],
        "gabarito": (
            "O usuário pode abrir a ficha e escrever uma informação "
            "adicional para encaminhamento à fila de revisão humana. "
            "O fluxo é simulado com informações fictícias."
        ),
    },
    {
        "id": "f007",
        "pergunta": (
            "Existe um número de WhatsApp do atendimento "
            "do Reencontro? Qual?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "f008",
        "pergunta": (
            "Em qual shopping já está instalado "
            "o totem do Reencontro?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "f009",
        "pergunta": (
            "Quantos dias leva para aprovar um pedido "
            "de cadastro no Reencontro?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "f010",
        "pergunta": (
            "Qual versão do aplicativo Reencontro "
            "está disponível na loja Android?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "f011",
        "pergunta": (
            "Quantas famílias já reencontraram alguém "
            "usando o projeto?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
    {
        "id": "f012",
        "pergunta": (
            "Qual empresa fabricou a câmera instalada "
            "no totem do projeto?"
        ),
        "fontes": [],
        "gabarito": "Essa informação não consta na base disponível.",
    },
]


def gerar(model, tokenizer, pergunta, documentos):
    mensagens = [
        {"role": "system", "content": INSTRUCAO},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "documentos": [
                        {
                            "fonte_id": item["id"],
                            "texto": item["texto"],
                        }
                        for item in documentos
                    ],
                    "pergunta": pergunta,
                },
                ensure_ascii=False,
            ),
        },
    ]

    entradas = tokenizer.apply_chat_template(
        mensagens,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    inicio = time.perf_counter()

    with torch.inference_mode():
        saida = model.generate(
            **entradas,
            max_new_tokens=350,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    novos_tokens = saida[0, entradas["input_ids"].shape[1]:]

    texto = tokenizer.decode(
        novos_tokens,
        skip_special_tokens=True,
    ).strip()

    segundos = time.perf_counter() - inicio
    dados, erro = validar_saida(texto, documentos)

    try:
        json.loads(texto)
        json_valido = True
    except json.JSONDecodeError:
        json_valido = False

    return {
        "saida_original": texto,
        "json_valido": json_valido,
        "validacao_formal_aprovada": erro is None,
        "erro_validacao": erro,
        "dados_validados": dados,
        "segundos_geracao": segundos,
        "tokens_gerados": len(novos_tokens),
        "avaliacao_humana": {
            "resposta_correta": None,
            "afirmacao_sem_suporte": None,
            "contradicao_com_fonte": None,
            "recusa_indevida": None,
            "observacoes": "",
        },
    }


def hash_arquivo(caminho):
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)

    pasta_resultados = PASTA / "avaliacao"
    pasta_resultados.mkdir(exist_ok=True)

    momento = datetime.now(timezone.utc)
    nome = momento.strftime("avaliacao_congelada_%Y%m%d_%H%M%S_%f.json")
    saida = pasta_resultados / nome

    busca = BuscaConhecimento(limiar=LIMIAR)

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

    arquivos = [
        PASTA / "base" / "conhecimento.json",
        PASTA / "busca.py",
        PASTA / "assistente_validado.py",
        ADAPTADOR / "adapter_config.json",
        ADAPTADOR / "adapter_model.safetensors",
        Path(__file__).resolve(),
    ]

    relatorio = {
        "data_utc": momento.isoformat(),
        "modelo": MODELO,
        "adaptador": ADAPTADOR.name,
        "limiar": LIMIAR,
        "instrucao": INSTRUCAO,
        "configuracao_geracao": {
            "do_sample": False,
            "max_new_tokens": 350,
            "enable_thinking": False,
        },
        "hashes_sha256": {
            str(arquivo.relative_to(PASTA)): hash_arquivo(arquivo)
            for arquivo in arquivos
            if arquivo.is_file()
        },
        "observacoes": [
            "Perguntas reformuladas sobre a mesma base do projeto.",
            "A mesma instrução documental é usada nas três configurações.",
            "Sem documentos, a instrução pode favorecer recusas.",
            "A geração é executada mesmo quando a busca retorna vazio.",
            "O bloqueio por busca vazia da API é avaliado separadamente.",
            "As marcações de correção semântica exigem leitura humana.",
        ],
        "resultados": [],
    }

    def salvar():
        saida.write_text(
            json.dumps(relatorio, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    salvar()

    for numero, caso in enumerate(CASOS, start=1):
        inicio_busca = time.perf_counter()
        documentos = busca.buscar(caso["pergunta"], quantidade=3)
        segundos_busca = time.perf_counter() - inicio_busca

        with model.disable_adapter():
            original = gerar(
                model, tokenizer, caso["pergunta"], []
            )
            rag = gerar(
                model, tokenizer, caso["pergunta"], documentos
            )

        especializado = gerar(
            model, tokenizer, caso["pergunta"], documentos
        )

        ids = [item["id"] for item in documentos]
        esperadas = set(caso["fontes"])

        recall_1 = None
        recall_3 = None

        if esperadas:
            recall_1 = len(esperadas.intersection(ids[:1])) / len(esperadas)
            recall_3 = len(esperadas.intersection(ids[:3])) / len(esperadas)

        relatorio["resultados"].append(
            {
                **caso,
                "documentos_recuperados": documentos,
                "segundos_busca": segundos_busca,
                "recall_1": recall_1,
                "recall_3": recall_3,
                "a_sem_contexto": original,
                "b_rag_simples": rag,
                "c_lora_rag": especializado,
            }
        )

        salvar()
        print(
            f"Teste {numero}/{len(CASOS)} concluído "
            f"| fontes: {ids}",
            flush=True,
        )

    resumo = {}

    for chave in ["a_sem_contexto", "b_rag_simples", "c_lora_rag"]:
        respostas = [
            item[chave] for item in relatorio["resultados"]
        ]

        resumo[chave] = {
            "json_validos": sum(
                item["json_valido"] for item in respostas
            ),
            "validacoes_formais_aprovadas": sum(
                item["validacao_formal_aprovada"]
                for item in respostas
            ),
            "total": len(respostas),
            "tempo_medio_geracao_segundos": sum(
                item["segundos_geracao"] for item in respostas
            ) / len(respostas),
        }

    for quantidade in [1, 3]:
        valores = [
            item[f"recall_{quantidade}"]
            for item in relatorio["resultados"]
            if item[f"recall_{quantidade}"] is not None
        ]
        resumo[f"recall_{quantidade}"] = sum(valores) / len(valores)

    relatorio["resumo_automatico"] = resumo
    salvar()

    print("\nAVALIAÇÃO CONCLUÍDA")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"\nArquivo salvo em: {saida}")


if __name__ == "__main__":
    main()