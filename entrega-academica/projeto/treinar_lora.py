import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELO = "Qwen/Qwen3-0.6B"
PASTA = Path(__file__).resolve().parent
SAIDA = PASTA / "artefatos" / "lora_experimento_02"

EPOCAS = 3
TAXA_APRENDIZADO = 0.0002

INSTRUCAO = (
    "Responda usando somente o contexto. "
    "Preserve negações e cite o ID da fonte entre colchetes. "
    "Se a informação não estiver no contexto, responda: "
    "Essa informação não consta na base disponível."
)


def carregar_dados(nome):
    caminho = PASTA / "dados" / f"{nome}.jsonl"

    return [
        json.loads(linha)
        for linha in caminho.read_text(encoding="utf-8-sig").splitlines()
        if linha.strip()
    ]


def preparar_exemplo(exemplo, tokenizer):
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

    prefixo = tokenizer.apply_chat_template(
        mensagens,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    ids_prefixo = tokenizer(
        prefixo,
        add_special_tokens=False,
    )["input_ids"]

    ids_resposta = tokenizer(
        exemplo["resposta"],
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    return {
        "input_ids": torch.tensor(
            [ids_prefixo + ids_resposta],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(
            (1, len(ids_prefixo) + len(ids_resposta)),
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [[-100] * len(ids_prefixo) + ids_resposta],
            dtype=torch.long,
        ),
    }


def validar(model, exemplos):
    model.eval()
    soma_perdas = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for exemplo in exemplos:
            resultado = model(**exemplo)
            quantidade = (exemplo["labels"][:, 1:] != -100).sum().item()

            soma_perdas += resultado.loss.item() * quantidade
            total_tokens += quantidade

    return soma_perdas / total_tokens


def main():
    if SAIDA.exists():
        raise FileExistsError(
            f"A pasta {SAIDA} já existe. "
            "Altere o nome de SAIDA para preservar o experimento anterior."
        )

    torch.set_num_threads(8)
    torch.manual_seed(42)

    print("Carregando o modelo...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODELO,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODELO,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    configuracao = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    # Inicia um adaptador novo, separado do teste técnico anterior.
    model = get_peft_model(model, configuracao)
    model.print_trainable_parameters()

    treino = [
        preparar_exemplo(item, tokenizer)
        for item in carregar_dados("treino")
    ]
    validacao = [
        preparar_exemplo(item, tokenizer)
        for item in carregar_dados("validacao")
    ]

    if not treino or not validacao:
        raise ValueError("Os arquivos de treino e validação devem ter exemplos.")

    parametros = [
        parametro
        for parametro in model.parameters()
        if parametro.requires_grad
    ]

    otimizador = torch.optim.AdamW(
        parametros,
        lr=TAXA_APRENDIZADO,
    )

    perda_inicial = validar(model, validacao)
    melhor_perda = perda_inicial
    melhor_epoca = 0
    historico = []

    SAIDA.mkdir(parents=True)
    tokenizer.save_pretrained(SAIDA)

    # Preserva a configuração inicial se nenhuma época melhorar a validação.
    model.save_pretrained(SAIDA)

    print(f"\nPerda inicial de validação: {perda_inicial:.4f}")
    inicio_total = time.perf_counter()

    for epoca in range(1, EPOCAS + 1):
        model.train()
        soma_perdas = 0.0
        total_tokens = 0
        inicio_epoca = time.perf_counter()

        ordem = torch.randperm(len(treino)).tolist()

        for passo, indice in enumerate(ordem, start=1):
            exemplo = treino[indice]
            otimizador.zero_grad(set_to_none=True)

            resultado = model(**exemplo)
            perda = resultado.loss

            if not torch.isfinite(perda).item():
                raise RuntimeError("Perda inválida durante o treinamento.")

            perda.backward()
            torch.nn.utils.clip_grad_norm_(parametros, max_norm=1.0)
            otimizador.step()

            quantidade = (exemplo["labels"][:, 1:] != -100).sum().item()
            soma_perdas += perda.item() * quantidade
            total_tokens += quantidade

            print(
                f"Época {epoca}/{EPOCAS} | "
                f"Passo {passo}/{len(treino)} | "
                f"Perda: {perda.item():.4f}",
                flush=True,
            )

        perda_treino = soma_perdas / total_tokens
        perda_validacao = validar(model, validacao)

        if perda_validacao < melhor_perda:
            melhor_perda = perda_validacao
            melhor_epoca = epoca
            model.save_pretrained(SAIDA)

        historico.append(
            {
                "epoca": epoca,
                "perda_treino": perda_treino,
                "perda_validacao": perda_validacao,
                "segundos": time.perf_counter() - inicio_epoca,
            }
        )

        print(
            f"\nResumo da época {epoca}: "
            f"treino={perda_treino:.4f} | "
            f"validação={perda_validacao:.4f}\n",
            flush=True,
        )

    relatorio = {
        "modelo": MODELO,
        "semente": 42,
        "epocas": EPOCAS,
        "taxa_aprendizado": TAXA_APRENDIZADO,
        "exemplos_treino": len(treino),
        "exemplos_validacao": len(validacao),
        "perda_validacao_inicial": perda_inicial,
        "melhor_epoca": melhor_epoca,
        "melhor_perda_validacao": melhor_perda,
        "historico": historico,
        "tempo_total_segundos": time.perf_counter() - inicio_total,
    }

    (SAIDA / "treinamento.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Treinamento concluído.")
    print(f"Melhor época: {melhor_epoca}")
    print(f"Adaptador selecionado salvo em: {SAIDA}")
    print("Época 0 significa que nenhuma época superou a validação inicial.")


if __name__ == "__main__":
    main()