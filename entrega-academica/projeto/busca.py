import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_BASE = PASTA_PROJETO / "base" / "conhecimento.json"


class BuscaConhecimento:
    def __init__(self, limiar=0.0):
        if not 0.0 <= limiar <= 1.0:
            raise ValueError("O limiar deve estar entre zero e um.")

        self.limiar = limiar

        self.trechos = json.loads(
            CAMINHO_BASE.read_text(encoding="utf-8-sig")
        )

        documentos = [
            f"{trecho['titulo']} {trecho['texto']}"
            for trecho in self.trechos
        ]

        self.vetorizador = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.matriz = self.vetorizador.fit_transform(documentos)

    def buscar(self, pergunta, quantidade=3):
        if not pergunta.strip() or quantidade <= 0:
            return []

        vetor_pergunta = self.vetorizador.transform([pergunta])

        similaridades = cosine_similarity(
            vetor_pergunta,
            self.matriz,
        )[0]

        indices = similaridades.argsort()[::-1][:quantidade]
        resultados = []

        for indice in indices:
            pontuacao = float(similaridades[indice])

            if pontuacao <= 0 or pontuacao < self.limiar:
                continue

            trecho = self.trechos[indice]

            resultados.append(
                {
                    "id": trecho["id"],
                    "titulo": trecho["titulo"],
                    "texto": trecho["texto"],
                    "fonte": trecho["fonte"],
                    "url": trecho["url"],
                    "pontuacao": pontuacao,
                }
            )

        return resultados


def main():
    busca = BuscaConhecimento(limiar=0.20)

    print("BUSCA NA BASE DO REENCONTRO")
    print(f"Limiar aplicado: {busca.limiar:.2f}")
    print("Digite uma pergunta ou 'sair' para encerrar.")

    while True:
        pergunta = input("\nSua pergunta: ").strip()

        if pergunta.lower() == "sair":
            break

        if not pergunta:
            continue

        resultados = busca.buscar(pergunta)

        if not resultados:
            print("Nenhum trecho atingiu o limiar da busca.")
            continue

        for posicao, resultado in enumerate(resultados, start=1):
            print(f"\nResultado {posicao}")
            print(f"ID: {resultado['id']}")
            print(f"Título: {resultado['titulo']}")
            print(f"Pontuação: {resultado['pontuacao']:.4f}")
            print(f"Trecho: {resultado['texto']}")
            print(f"Fonte: {resultado['fonte']}")


if __name__ == "__main__":
    main()