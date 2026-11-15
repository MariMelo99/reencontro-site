import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-0.6B"
PASTA = Path(__file__).resolve().parent
ADAPTADOR = PASTA / "artefatos" / "lora_experimento_01"
SAIDA = ADAPTADOR / "comparacao_teste.json"

INSTRUCAO = (
    "Responda usando somente o contexto. "
    "Preserve negações e cite o ID da fonte entre colchetes. "
    "Se a informação não estiver no contexto, responda: "
    "Essa informação não consta na base disponível."
)


def gerar(model, tokenizer, exemplo):
    mensagens = [
        {"role": "system", "content": INSTRUCAO},
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
    }


def main():
    if SAIDA.exists():
        raise FileExistsError(
            "A comparação já foi salva. Preserve o resultado anterior "
            "e altere o nome de SAIDA para uma nova execução."
        )

    torch.set_num_threads(8)
    torch.manual_seed(42)

    testes = [
        json.loads(linha)
        for linha in (PASTA / "dados" / "teste.jsonl").read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if linha.strip()
    ]

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

    model = PeftModel.from_pretrained(base, ADAPTADOR)
    model.config.use_cache = True
    model.eval()

    resultados = []

    for exemplo in testes:
        # Usa o mesmo modelo e contexto, desativando somente o LoRA.
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

        print(f"\nPERGUNTA: {exemplo['pergunta']}")
        print(f"GABARITO: {exemplo['resposta']}")
        print(f"ORIGINAL: {original['resposta']}")
        print(f"COM LORA: {especializado['resposta']}", flush=True)

    SAIDA.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nComparação salva em: {SAIDA}")


if __name__ == "__main__":
    main()