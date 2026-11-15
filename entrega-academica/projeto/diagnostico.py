import torch
from pathlib import Path
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-0.6B"

torch.set_num_threads(8)

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Carregando:", MODELO)

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
model.eval()

testes = [
    (
        "Entrada gratuita",
        "Documento: Não é necessário comprar ingresso para entrar "
        "na exposição.\n"
        "Pergunta: Preciso comprar ingresso para entrar?\n"
        "Responda usando somente o documento.",
    ),
    (
        "Serviço indisponível",
        "Documento: A biblioteca não abre aos domingos.\n"
        "Pergunta: A biblioteca abre aos domingos?\n"
        "Responda usando somente o documento.",
    ),
    (
        "Informação ausente",
        "Documento: A biblioteca abre aos sábados.\n"
        "Pergunta: Qual é o telefone da biblioteca?\n"
        "Se a informação não estiver no documento, responda NÃO CONSTA.",
    ),
    (
        "Extração literal",
        "Documento: Não é necessário esperar 24 horas para "
        "comunicar o desaparecimento.\n"
        "Copie o documento inteiro, sem alterar nenhuma palavra.",
    ),
]

configuracoes = [
    ("Determinística", {"do_sample": False}),
    (
        "Amostragem sugerida",
        {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        },
    ),
]
registro = []

for nome_teste, pergunta in testes:
    mensagens = [{"role": "user", "content": pergunta}]

    entradas = tokenizer.apply_chat_template(
        mensagens,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    print(f"\nTESTE: {nome_teste}")
    registro.append(f"\nTESTE: {nome_teste}")

    for nome_configuracao, parametros in configuracoes:
        torch.manual_seed(42)

        with torch.inference_mode():
            saida = model.generate(
                **entradas,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                **parametros,
            )

        novos_tokens = saida[0, entradas["input_ids"].shape[1]:]
        resposta = tokenizer.decode(
            novos_tokens,
            skip_special_tokens=True,
        ).strip()

        print(f"\n{nome_configuracao}:")
        print(resposta)
        registro.append(f"\n{nome_configuracao}:\n{resposta}")

print("\nDiagnóstico concluído.")
Path("resultado_diagnostico.txt").write_text(
    "\n".join(registro),
    encoding="utf-8",
)