import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-0.6B"
PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_SAIDA = PASTA_PROJETO / "artefatos" / "lora_teste"

torch.set_num_threads(8)
torch.manual_seed(42)

exemplos = [
    {
        "contexto": "[treino_001] Não é necessário comprar ingresso.",
        "pergunta": "Preciso comprar ingresso?",
        "resposta": (
            "Não é necessário comprar ingresso. [treino_001]"
        ),
    },
    {
        "contexto": "[treino_002] É necessário apresentar o comprovante.",
        "pergunta": "Preciso apresentar o comprovante?",
        "resposta": (
            "É necessário apresentar o comprovante. [treino_002]"
        ),
    },
    {
        "contexto": "[treino_003] A unidade não atende aos domingos.",
        "pergunta": "A unidade atende aos domingos?",
        "resposta": (
            "A unidade não atende aos domingos. [treino_003]"
        ),
    },
    {
        "contexto": "[treino_004] A unidade atende às segundas-feiras.",
        "pergunta": "Qual é o telefone da unidade?",
        "resposta": "Essa informação não consta na base disponível.",
    },
]

print("Carregando o modelo para o teste de treinamento...")

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

model = get_peft_model(model, configuracao)
model.print_trainable_parameters()
model.train()

otimizador = torch.optim.AdamW(
    [parametro for parametro in model.parameters()
     if parametro.requires_grad],
    lr=0.0002,
)

historico = []
inicio_total = time.perf_counter()

for passo in range(5):
    exemplo = exemplos[passo % len(exemplos)]

    mensagens = [
        {
            "role": "system",
            "content": (
                "Responda usando somente o contexto. Cite o ID da fonte. "
                "Se a resposta não estiver disponível, informe que "
                "ela não consta na base."
            ),
        },
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

    # Treina somente a resposta, preservando o prompt como contexto.
    ids_prefixo = tokenizer(
        prefixo,
        add_special_tokens=False,
    )["input_ids"]

    ids_resposta = tokenizer(
        exemplo["resposta"],
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    input_ids = torch.tensor(
        [ids_prefixo + ids_resposta],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [[-100] * len(ids_prefixo) + ids_resposta],
        dtype=torch.long,
    )

    inicio_passo = time.perf_counter()
    otimizador.zero_grad(set_to_none=True)

    resultado = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )

    perda = resultado.loss

    if not torch.isfinite(perda).item():
        raise RuntimeError("Perda inválida. Treinamento interrompido.")

    perda.backward()
    torch.nn.utils.clip_grad_norm_(
        [parametro for parametro in model.parameters()
         if parametro.requires_grad],
        max_norm=1.0,
    )
    otimizador.step()

    segundos = time.perf_counter() - inicio_passo

    historico.append(
        {
            "passo": passo + 1,
            "perda": perda.item(),
            "segundos": segundos,
        }
    )

    print(
        f"Passo {passo + 1}/5 | "
        f"Perda: {perda.item():.4f} | "
        f"Tempo: {segundos:.1f}s",
        flush=True,
    )

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
model.save_pretrained(PASTA_SAIDA)
tokenizer.save_pretrained(PASTA_SAIDA)

relatorio = {
    "modelo": MODELO,
    "finalidade": "Teste técnico de treinamento, sem avaliação de qualidade",
    "historico": historico,
    "tempo_total_segundos": time.perf_counter() - inicio_total,
}

(PASTA_SAIDA / "teste_treino.json").write_text(
    json.dumps(relatorio, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nTeste concluído.")
print(f"Adaptador e relatório salvos em: {PASTA_SAIDA}")