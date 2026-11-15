import ast
import hashlib
import json
from pathlib import Path

PASTA = Path(__file__).resolve().parent


def main():
    for arquivo in PASTA.glob("*.py"):
        ast.parse(arquivo.read_text(encoding="utf-8-sig"), filename=arquivo.name)
    print("Sintaxe dos scripts Python válida.")
    base = json.loads((PASTA / "base/conhecimento.json").read_text(encoding="utf-8-sig"))
    assert len(base) == 7
    assert len({item["id"] for item in base}) == 7
    for nome, esperado in [("treino_json.json", 32), ("validacao_json.json", 8)]:
        dados = json.loads((PASTA / "artefatos/lora_experimento_03" / nome).read_text(encoding="utf-8"))
        assert len(dados) == esperado
    manifesto = json.loads((PASTA / "docs/integridade_arquivos.json").read_text(encoding="utf-8"))
    divergencias = []
    for nome, esperado in manifesto.items():
        arquivo = PASTA / nome
        if not arquivo.is_file() or hashlib.sha256(arquivo.read_bytes()).hexdigest() != esperado:
            divergencias.append(nome)
    if divergencias:
        raise RuntimeError("Arquivos diferentes do pacote organizado: " + ", ".join(divergencias))
    print("Base e conjuntos conferidos. Integridade dos arquivos do manifesto confirmada.")
    print("Esta verificação não executa o modelo nem substitui os testes da API.")


if __name__ == "__main__":
    main()
