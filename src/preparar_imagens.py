# -*- coding: utf-8 -*-
"""
CardioIA, Fase 2, Ir Alem 2: preparacao das imagens de ECG.

O dataset recomendado pelo enunciado (Kaggle, shayanfazeli/heartbeat) traz os
batimentos como series numericas de 187 amostras, e nao como imagens. Este
modulo converte cada batimento em uma imagem em tons de cinza (o tracado do
batimento desenhado em um quadro quadrado), grava exemplos em PNG e carrega as
imagens de volta no formato que a rede neural espera (redimensionadas,
em tons de cinza, normalizadas e achatadas em vetor).

Execucao direta (gera os exemplos de imagens do repositorio):

    python src/preparar_imagens.py

As mesmas funcoes estao definidas nas celulas do notebook
src/notebooks/cardioia_ecg_mlp.ipynb, que e autossuficiente e nao importa
este arquivo; este modulo e o utilitario de linha de comando equivalente.

Dependencias: numpy, pandas, pillow.

Aviso: projeto academico com dados publicos anonimizados; nao e ferramenta de
diagnostico e nao substitui a avaliacao de um profissional de saude.
"""

import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DATASET = os.path.join(RAIZ, "data")
PASTA_EXEMPLOS = os.path.join(RAIZ, "assets", "exemplos_imagens")

ARQUIVOS = {
    "ptbdb_normal.csv": "033d355e76e9f65f692fa398c5b24e357d004fab328c97db5160591a2eafe48c",
    "ptbdb_abnormal.csv": "27b4a37cd6dbcede0900cde3df02291f21e7538085d54e6b33abb91f4a5cf912",
}
# Espelho publico usado como alternativa quando os arquivos do Kaggle nao estao na pasta.
URL_ESPELHO = "https://raw.githubusercontent.com/jorgesandoval/heartbeat-classification-cnn/main/data/"

N_AMOSTRAS = 187          # amostras por batimento (125 Hz, preenchidas com zero ate 187)
LADO_IMAGEM = 96          # lado da imagem gerada (pixels)
LADO_REDE = 32            # lado apos o redimensionamento para a rede
ESPESSURA = 2             # espessura do traco em pixels


def sha256_arquivo(caminho):
    """Calcula o SHA-256 de um arquivo (para conferir a integridade do download)."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def localizar_dataset(pasta=PASTA_DATASET, baixar_se_ausente=True):
    """Garante que os dois arquivos do subconjunto PTB existem na pasta.

    Procura os arquivos (baixados manualmente do Kaggle); se faltarem e
    `baixar_se_ausente` for True, obtem-nos do espelho publico. Confere o
    SHA-256 e devolve um dicionario nome -> caminho.
    """
    os.makedirs(pasta, exist_ok=True)
    caminhos = {}
    for nome, hash_esperado in ARQUIVOS.items():
        caminho = os.path.join(pasta, nome)
        if not os.path.exists(caminho):
            if not baixar_se_ausente:
                raise FileNotFoundError(f"{nome} nao encontrado em {pasta}. Baixe o dataset do Kaggle (ver README).")
            print(f"Baixando {nome} do espelho publico...")
            urllib.request.urlretrieve(URL_ESPELHO + nome, caminho)
        hash_obtido = sha256_arquivo(caminho)
        if hash_obtido != hash_esperado:
            print(f"Atencao: SHA-256 de {nome} difere do registrado ({hash_obtido[:12]}... vs {hash_esperado[:12]}...).")
        caminhos[nome] = caminho
    return caminhos


def carregar_ptb(pasta=PASTA_DATASET):
    """Le os CSVs do PTB e devolve (X, y): X com shape (n, 187) em float32 e
    y com 0 = normal e 1 = anormal."""
    caminhos = localizar_dataset(pasta)
    normal = pd.read_csv(caminhos["ptbdb_normal.csv"], header=None).to_numpy(dtype=np.float32)
    anormal = pd.read_csv(caminhos["ptbdb_abnormal.csv"], header=None).to_numpy(dtype=np.float32)
    dados = np.vstack([normal, anormal])
    X = dados[:, :N_AMOSTRAS]
    y = dados[:, N_AMOSTRAS].astype(np.int64)
    return X, y


def batimentos_para_imagens(X, lado=LADO_IMAGEM, espessura=ESPESSURA):
    """Converte um lote de batimentos (n, 187) em imagens em tons de cinza
    (n, lado, lado) com valores 0 (fundo) e 255 (traco).

    O tracado e desenhado como em um grafico de linha: cada coluna da imagem
    recebe um segmento vertical entre a amplitude da coluna anterior e a da
    coluna atual, o que mantem a linha continua. Operacao vetorizada por
    coluna, para que o lote inteiro seja convertido em poucos segundos.
    """
    n = X.shape[0]
    posicoes = np.linspace(0, X.shape[1] - 1, lado)
    indices = np.arange(X.shape[1])
    # Interpola a amplitude de cada batimento nas `lado` colunas da imagem
    amplitudes = np.vstack([np.interp(posicoes, indices, sinal) for sinal in X])
    linhas = np.rint((1.0 - np.clip(amplitudes, 0.0, 1.0)) * (lado - 1)).astype(np.int32)  # amplitude alta = topo
    # O dataset preenche o fim de cada batimento com zeros ate 187 amostras; o traco
    # termina na ultima amostra nao nula, de modo que a duracao do batimento fica
    # visivel na imagem em vez de virar uma linha reta no fundo.
    ultima_amostra = np.array([np.flatnonzero(sinal > 0).max() if np.any(sinal > 0) else 0 for sinal in X])
    ultima_coluna = np.floor(ultima_amostra / (X.shape[1] - 1) * (lado - 1)).astype(np.int32)
    imagens = np.zeros((n, lado, lado), dtype=np.uint8)
    grade = np.arange(lado)[None, :]
    for coluna in range(lado):
        anterior = linhas[:, max(coluna - 1, 0)]
        atual = linhas[:, coluna]
        inferior = np.minimum(anterior, atual)[:, None]
        superior = np.maximum(anterior, atual)[:, None]
        ativo = (coluna <= ultima_coluna)[:, None]
        mascara = (grade >= inferior) & (grade <= superior) & ativo
        imagens[:, :, coluna] = np.where(mascara, 255, 0)
    # Engrossa o traco deslocando a imagem uma linha para baixo e combinando
    for _ in range(max(espessura - 1, 0)):
        deslocada = np.zeros_like(imagens)
        deslocada[:, 1:, :] = imagens[:, :-1, :]
        imagens = np.maximum(imagens, deslocada)
    return imagens


def salvar_exemplos(imagens, y, pasta=PASTA_EXEMPLOS, por_classe=10):
    """Grava `por_classe` imagens PNG de cada classe (os primeiros batimentos
    de cada rotulo) em `pasta`. Devolve a lista de caminhos gravados."""
    os.makedirs(pasta, exist_ok=True)
    caminhos = []
    for rotulo, nome in [(0, "normal"), (1, "anormal")]:
        indices = np.flatnonzero(y == rotulo)[:por_classe]
        for k, i in enumerate(indices, 1):
            caminho = os.path.join(pasta, f"{nome}_{k:02d}.png")
            Image.fromarray(imagens[i], mode="L").save(caminho)
            caminhos.append(caminho)
    return caminhos


def imagem_para_vetor(imagem, lado=LADO_REDE):
    """Pre-processamento de uma imagem para a rede: converte para tons de
    cinza ('L'), redimensiona para lado x lado, normaliza para [0, 1] e achata.
    Aceita um array numpy ou um objeto PIL.Image."""
    if not isinstance(imagem, Image.Image):
        imagem = Image.fromarray(np.asarray(imagem, dtype=np.uint8))
    imagem = imagem.convert("L").resize((lado, lado), Image.Resampling.BILINEAR)
    return (np.asarray(imagem, dtype=np.float32) / 255.0).reshape(-1)


def imagens_para_vetores(imagens, lado=LADO_REDE):
    """Aplica `imagem_para_vetor` a um lote (n, altura, largura) e devolve (n, lado*lado)."""
    return np.vstack([imagem_para_vetor(img, lado) for img in imagens])


def carregar_pasta_de_imagens(pasta, lado=LADO_REDE):
    """Carrega todos os PNGs de uma pasta como vetores (uso ilustrativo: mostra
    que o pipeline funciona a partir de arquivos de imagem, nao so de arrays)."""
    nomes = sorted(n for n in os.listdir(pasta) if n.lower().endswith(".png"))
    vetores = [imagem_para_vetor(Image.open(os.path.join(pasta, n)), lado) for n in nomes]
    rotulos = np.array([0 if n.startswith("normal") else 1 for n in nomes])
    return np.vstack(vetores), rotulos, nomes


if __name__ == "__main__":
    X, y = carregar_ptb()
    print(f"Batimentos: {X.shape[0]} ({int((y == 0).sum())} normais, {int((y == 1).sum())} anormais), {X.shape[1]} amostras cada.")
    selecao = np.concatenate([np.flatnonzero(y == 0)[:10], np.flatnonzero(y == 1)[:10]])
    imagens = batimentos_para_imagens(X[selecao])
    gravados = salvar_exemplos(imagens, y[selecao])
    print(f"{len(gravados)} exemplos gravados em {os.path.relpath(PASTA_EXEMPLOS, RAIZ)}/")
    vetores = imagens_para_vetores(imagens[:5])
    print(f"Vetores de entrada da rede: shape {vetores.shape}, valores entre {vetores.min():.2f} e {vetores.max():.2f}")
