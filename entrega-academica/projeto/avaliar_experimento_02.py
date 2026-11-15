import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELO = "Qwen/Qwen3-0.6B"
PASTA = Path(__file__).resolve().parent
ADAPTADOR = PASTA / "artefatos" / "lora_experimento_02"
SAIDA = ADAPTADOR / "comparacao_novos_cenarios.json"

INSTRUCAO = (
    "Responda usando somente o contexto. "
    "Preserve negações e cite o ID da fonte entre colchetes. "
    "Se a informação não estiver no contexto, responda: "
    "Essa informação não consta na base disponível."
)

# Cenários fictícios de avaliação.
# Estes exemplos não devem ser acrescentados ao treinamento.
TESTES = [
    {
        "id": "novo_001",
        "contexto": (
            "[novo_001] Não é necessário criar uma conta "
            "para consultar o catálogo."
        ),
        "pergunta": (
            "Preciso criar uma conta para consultar o catálogo?"
        ),
        "resposta": (
            "Não é necessário criar uma conta para consultar "
            "o catálogo. [novo_001]"
        ),
    },
    {
        "id": "novo_002",
        "contexto": (
            "[novo_002] É necessário reservar um horário "
            "para utilizar o laboratório."
        ),
        "pergunta": (
            "Preciso reservar um horário para usar o laboratório?"
        ),
        "resposta": (
            "É necessário reservar um horário para utilizar "
            "o laboratório. [novo_002]"
        ),
    },
    {
        "id": "novo_003",
        "contexto": (
            "[novo_003] A inscrição na atividade não garante "
            "o recebimento de certificado."
        ),
        "pergunta": (
            "A inscrição garante que receberei um certificado?"
        ),
        "resposta": (
            "A inscrição na atividade não garante o recebimento "
            "de certificado. [novo_003]"
        ),
    },
    {
        "id": "novo_004",
        "contexto": (
            "[novo_004] O museu oferece visitas guiadas "
            "durante a semana."
        ),
        "pergunta": "Qual é o preço da visita guiada?",
        "resposta": "Essa informação não consta na base disponível.",
    },
    {
        "id": "novo_005",
        "contexto": (
            "[novo_005] A solicitação foi recebida "
            "pelo setor responsável."
        ),
        "pergunta": (
            "Em qual data a solicitação será concluída?"
        ),
        "resposta": "Essa informação não consta na base disponível.",
    },
    {
        "id": "novo_006",
        "contexto": (
            "[novo_006] O cadastro do equipamento contém "
            "a marca e o número de série."
        ),
        "pergunta": (
            "Qual é a localização atual do equipamento?"
        ),
        "resposta": "Essa informação não consta na base disponível.",
    },
]


def gerar(model, tokenizer, exemplo):
    mensagens = [
        {
            "role": "system",
            "content": INSTRUCAO,
        },
        {
            "role": "user",
            "content": (
                f"Contexto:\n{exemplo['contexto']}\n\n"
                f"Pergunta:\n{exemplo['pergunta']}"
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
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    novos_tokens = saida[0, entradas["input_ids"].shape[1]:]

    return {
        "resposta": tokenizer.decode(
            novos_tokens,
            skip_special_tokens=True,
        ).strip(),
        "segundos": time.perf_counter() - inicio,
        "tokens_gerados": len(novos_tokens),
    }


def main():
    if not (ADAPTADOR / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Adaptador não encontrado em: {ADAPTADOR}"
        )

    if SAIDA.exists():
        raise FileExistsError(
            f"O resultado já existe em: {SAIDA}. "
            "Altere o nome de SAIDA para preservar a execução anterior."
        )

    torch.set_num_threads(8)
    torch.manual_seed(42)

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

    for numero, exemplo in enumerate(TESTES, start=1):
        # Mantém contexto e geração iguais, alterando apenas o adaptador.
        with model.disable_adapter():
            original = gerar(model, tokenizer, exemplo)

        especializado = gerar(model, tokenizer, exemplo)

        registro = {
            "id": exemplo["id"],
            "contexto": exemplo["contexto"],
            "pergunta": exemplo["pergunta"],
            "gabarito": exemplo["resposta"],
            "original_com_contexto": original,
            "lora_com_contexto": especializado,
        }
        resultados.append(registro)

        print(f"\nTESTE {numero}/{len(TESTES)}")
        print(f"PERGUNTA: {exemplo['pergunta']}")
        print(f"GABARITO: {exemplo['resposta']}")
        print(f"ORIGINAL: {original['resposta']}")
        print(f"COM LORA: {especializado['resposta']}")
        print(
            f"TEMPO ORIGINAL: {original['segundos']:.1f}s | "
            f"TEMPO LORA: {especializado['segundos']:.1f}s",
            flush=True,
        )

    SAIDA.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nComparação salva em: {SAIDA}")


if __name__ == "__main__":
    main()