import spaces
import json
import torch
import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from assistente_validado import MODELO, ADAPTADOR, INSTRUCAO, validar_saida
from busca import BuscaConhecimento

torch.manual_seed(42)
busca = BuscaConhecimento(limiar=0.20)
tokenizer = AutoTokenizer.from_pretrained(MODELO)
base = AutoModelForCausalLM.from_pretrained(
    MODELO, torch_dtype=torch.float32, attn_implementation="eager"
)
model = PeftModel.from_pretrained(
    base,
    ADAPTADOR,
    is_trainable=False,
    torch_device="cpu",
)
model.eval()
model.to("cuda")

@spaces.GPU(duration=45)
def gerar(mensagens):
    entradas = tokenizer.apply_chat_template(
        mensagens, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        saida = model.generate(
            **entradas, max_new_tokens=350, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(
        saida[0, entradas["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

def consultar(pergunta):
    if not isinstance(pergunta, str) or not 3 <= len(pergunta.strip()) <= 600:
        raise gr.Error("Escreva uma pergunta entre 3 e 600 caracteres.")
    pergunta = pergunta.strip().rstrip("?!.,;:…").rstrip()
    if len(pergunta) < 3:
        raise gr.Error("Escreva uma pergunta com pelo menos 3 caracteres.")
    recuperados = busca.buscar(pergunta, quantidade=3)
    if not recuperados:
        return dict(status="nao_consta", resposta="Essa informação não consta na base disponível.",
                    evidencias=[], validacao="formal_aprovada")
    mensagens = [
        {"role": "system", "content": INSTRUCAO},
        {"role": "user", "content": json.dumps({
            "documentos": [{"fonte_id": t["id"], "texto": t["texto"]} for t in recuperados],
            "pergunta": pergunta.strip()
        }, ensure_ascii=False)}
    ]
    texto = gerar(mensagens)
    dados, erro = validar_saida(texto, recuperados)
    if erro:
        return dict(status="bloqueado", resposta="Não foi possível validar a resposta gerada. Procure orientação humana.",
                    evidencias=[], validacao="reprovada")
    fontes = {t["id"]: t for t in recuperados}
    return dict(status=dados["status"], resposta=dados["resposta"],
                evidencias=[{**e, "fonte": fontes[e["fonte_id"]]["fonte"],
                             "url": fontes[e["fonte_id"]]["url"]} for e in dados["evidencias"]],
                validacao="formal_aprovada")

with gr.Blocks(title="Assistente Reencontro") as demo:
    gr.Markdown("# Assistente Reencontro\nDemonstração acadêmica FIAP CitizenTech. Use perguntas fictícias, sem dados pessoais. A IA pode errar; confira as fontes. O serviço gratuito está sujeito a fila e cota.")
    pergunta = gr.Textbox(label="Sua pergunta", max_length=600, lines=3)
    enviar = gr.Button("Enviar pergunta", variant="primary")
    resultado = gr.JSON(label="Resposta e evidências")
    enviar.click(consultar, inputs=pergunta, outputs=resultado, api_name="consultar", concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
