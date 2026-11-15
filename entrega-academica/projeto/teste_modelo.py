import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-0.6B"

torch.set_num_threads(8)
torch.manual_seed(42)

print("Carregando o modelo. Na primeira execução haverá um download.")

tokenizer = AutoTokenizer.from_pretrained(MODELO)

model = AutoModelForCausalLM.from_pretrained(
    MODELO,
    dtype=torch.float32,
)
model.to("cpu")
model.eval()

mensagens = [
    {
        "role": "system",
        "content": "Responda em português, de forma breve e clara.",
    },
    {
        "role": "user",
        "content": (
            "Como organizar uma lista de tarefas para desenvolver "
            "um projeto acadêmico?"
        ),
    },
]

texto = tokenizer.apply_chat_template(
    mensagens,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

entradas = tokenizer(texto, return_tensors="pt")

print("Gerando uma resposta pela CPU...")
inicio = time.perf_counter()

with torch.inference_mode():
    saida = model.generate(
        **entradas,
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

novos_tokens = saida[0, entradas["input_ids"].shape[1]:]
resposta = tokenizer.decode(novos_tokens, skip_special_tokens=True)
duracao = time.perf_counter() - inicio

print("\nRESPOSTA DO MODELO")
print(resposta)
print(f"\nTempo de geração: {duracao:.1f} segundos")
print(f"Tokens gerados: {len(novos_tokens)}")