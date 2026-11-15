from huggingface_hub import snapshot_download

if __name__ == "__main__":
    caminho = snapshot_download(repo_id="Qwen/Qwen3-0.6B")
    print("Modelo disponível no cache local:")
    print(caminho)
    print("Registre a revisão do snapshot para novas reproduções.")
