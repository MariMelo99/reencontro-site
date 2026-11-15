import json
from contextlib import asynccontextmanager
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from assistente_validado import ADAPTADOR, INSTRUCAO, MODELO, validar_saida
from busca import BuscaConhecimento


class Consulta(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    pergunta: str = Field(min_length=3, max_length=600)


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(8)
    torch.manual_seed(42)
    print("Carregando o assistente Reencontro...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODELO,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, ADAPTADOR, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.busca = BuscaConhecimento(limiar=0.20)
    app.state.trava = Lock()
    print("Assistente carregado.")
    try:
        yield
    finally:
        del app.state.model
        del app.state.tokenizer
        del app.state.busca


app = FastAPI(
    title="Reencontro — API local de demonstração",
    description="Use perguntas fictícias sem dados pessoais.",
    version="0.2.0",
    lifespan=lifespan,
)

# O navegador poderá consultar esta API a partir do site local na porta 5500.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@app.get("/saude")
def saude():
    return {
        "status": "disponivel",
        "modelo": MODELO,
        "adaptador": ADAPTADOR.name,
    }


@app.post("/consultar")
def consultar(consulta: Consulta):
    if not app.state.trava.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="O assistente está ocupado. Tente novamente em instantes.",
        )
    try:
        recuperados = app.state.busca.buscar(consulta.pergunta, quantidade=3)
        if not recuperados:
            return {
                "status": "nao_consta",
                "resposta": "Essa informação não consta na base disponível.",
                "evidencias": [],
                "validacao": "formal_aprovada",
                "origem": "filtro_de_recuperacao",
                "aviso": (
                    "Nenhum trecho atingiu o limiar da busca. "
                    "Isso não comprova que a informação inexiste."
                ),
            }
        documentos = [
            {"fonte_id": trecho["id"], "texto": trecho["texto"]}
            for trecho in recuperados
        ]
        mensagens = [
            {"role": "system", "content": INSTRUCAO},
            {
                "role": "user",
                "content": json.dumps(
                    {"documentos": documentos, "pergunta": consulta.pergunta},
                    ensure_ascii=False,
                ),
            },
        ]
        entradas = app.state.tokenizer.apply_chat_template(
            mensagens,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            saida = app.state.model.generate(
                **entradas,
                max_new_tokens=350,
                do_sample=False,
                pad_token_id=app.state.tokenizer.eos_token_id,
            )
        novos_tokens = saida[0, entradas["input_ids"].shape[1]:]
        texto = app.state.tokenizer.decode(
            novos_tokens, skip_special_tokens=True
        ).strip()
        dados, erro = validar_saida(texto, recuperados)
        if erro:
            return {
                "status": "bloqueado",
                "resposta": (
                    "Não foi possível validar a resposta gerada. "
                    "Procure orientação humana."
                ),
                "evidencias": [],
                "validacao": "reprovada",
            }
        fontes = {trecho["id"]: trecho for trecho in recuperados}
        evidencias = [
            {
                **evidencia,
                "fonte": fontes[evidencia["fonte_id"]]["fonte"],
                "url": fontes[evidencia["fonte_id"]]["url"],
            }
            for evidencia in dados["evidencias"]
        ]
        return {
            "status": dados["status"],
            "resposta": dados["resposta"],
            "evidencias": evidencias,
            "validacao": "formal_aprovada",
            "aviso": (
                "Demonstração acadêmica. A validação formal não garante "
                "a pertinência nem a preservação do sentido."
            ),
        }
    finally:
        app.state.trava.release()
