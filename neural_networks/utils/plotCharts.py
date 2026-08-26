"""
Geracao de graficos a partir dos CSVs produzidos por neural_networks.utils.Statistics.

Uso, a partir do diretorio neural_networks/:

    python utils/plotCharts.py individual "NeuralNetwork"
    python utils/plotCharts.py comparativo "NeuralNetwork" "MLP"

O modo 'individual' plota o CSV mais recente de data/. O modo 'comparativo' plota os
dois mais recentes lado a lado, assumindo que o mais recente e' o primeiro rotulo.
As imagens vao para img/, criado se nao existir.

Colunas esperadas no CSV:
    epoch, train_error, train_classification_errors,
    validation_error, validation_classification_errors,
    test_error, test_classification_errors

A coluna de validacao fica vazia quando o treinamento roda sem conjunto de validacao.

Dependencias: pandas, matplotlib.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# Diretorios resolvidos a partir da localizacao deste arquivo, e nao do diretorio
# atual: assim o script funciona de onde quer que seja chamado.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_CSV = os.path.join(RAIZ, "data")
DIR_IMG = os.path.join(RAIZ, "img")


def csvs_mais_recentes(diretorio, quantidade):
    """Devolve os N arquivos .csv mais recentes de um diretorio, do mais novo ao mais antigo."""
    arquivos = glob.glob(os.path.join(diretorio, "*.csv"))
    if len(arquivos) < quantidade:
        return []
    return sorted(arquivos, key=os.path.getmtime, reverse=True)[:quantidade]


def carregar(caminho):
    """Le um CSV de estatisticas e devolve o DataFrame, ou None em caso de erro."""
    try:
        return pd.read_csv(caminho, sep=",")
    except Exception as erro:
        print(f"Erro ao ler '{caminho}': {erro}")
        return None


def salvar(fig, nome):
    """Salva a figura em img/ e fecha para liberar memoria."""
    os.makedirs(DIR_IMG, exist_ok=True)
    caminho = os.path.join(DIR_IMG, nome)
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico salvo em '{caminho}'")


def tem_validacao(df):
    """Informa se o CSV traz uma coluna de validacao preenchida."""
    return "validation_error" in df.columns and df["validation_error"].notna().any()


def plotar_individual(caminho_csv, nome_modelo):
    """Gera os graficos de erro e de classificacoes incorretas de uma unica execucao."""
    df = carregar(caminho_csv)
    if df is None:
        return

    print(f"Gerando graficos individuais para: {nome_modelo}")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df["epoch"], df["train_error"], label="Treino", color="#1f77b4")
    if tem_validacao(df):
        ax.plot(df["epoch"], df["validation_error"], label="Validacao", color="#2ca02c")
        melhor = df.loc[df["validation_error"].idxmin()]
        ax.axvline(melhor["epoch"], color="#2ca02c", linestyle=":", linewidth=1.2)
        ax.annotate(
            f"melhor validacao\nepoca {int(melhor['epoch'])}",
            xy=(melhor["epoch"], melhor["validation_error"]),
            xytext=(8, 12), textcoords="offset points", fontsize=9, color="#2ca02c",
        )
    ax.plot(df["epoch"], df["test_error"], label="Teste", color="#ff7f0e", linestyle="--")
    ax.set_title(f"Erro medio por epoca ({nome_modelo})", fontsize=15)
    ax.set_xlabel("Epoca", fontsize=12)
    ax.set_ylabel("Erro medio por amostra", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    salvar(fig, f"erro_{nome_modelo}.png")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df["epoch"], df["train_classification_errors"], label="Treino", color="#1f77b4")
    if tem_validacao(df):
        ax.plot(df["epoch"], df["validation_classification_errors"], label="Validacao", color="#2ca02c")
    ax.plot(df["epoch"], df["test_classification_errors"], label="Teste", color="#d62728", linestyle="--")
    ax.set_title(f"Classificacoes incorretas por epoca ({nome_modelo})", fontsize=15)
    ax.set_xlabel("Epoca", fontsize=12)
    ax.set_ylabel("Amostras classificadas incorretamente", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    salvar(fig, f"erros_classificacao_{nome_modelo}.png")


def plotar_comparativo(csv_a, csv_b, nome_a, nome_b):
    """Compara duas execucoes no mesmo par de graficos."""
    df_a = carregar(csv_a)
    df_b = carregar(csv_b)
    if df_a is None or df_b is None:
        return

    print(f"Gerando graficos comparativos: {nome_a} x {nome_b}")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df_a["epoch"], df_a["train_error"], label=f"Treino ({nome_a})", color="#1f77b4")
    ax.plot(df_a["epoch"], df_a["test_error"], label=f"Teste ({nome_a})", color="#7fbfe0", linestyle="--")
    ax.plot(df_b["epoch"], df_b["train_error"], label=f"Treino ({nome_b})", color="#d62728")
    ax.plot(df_b["epoch"], df_b["test_error"], label=f"Teste ({nome_b})", color="#f0a1a0", linestyle="--")
    ax.set_title("Comparativo de erro medio por epoca", fontsize=15)
    ax.set_xlabel("Epoca", fontsize=12)
    ax.set_ylabel("Erro medio por amostra", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    salvar(fig, "comparativo_erro.png")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df_a["epoch"], df_a["train_classification_errors"], label=f"Treino ({nome_a})", color="#2ca02c")
    ax.plot(df_a["epoch"], df_a["test_classification_errors"], label=f"Teste ({nome_a})", color="#98df8a", linestyle="--")
    ax.plot(df_b["epoch"], df_b["train_classification_errors"], label=f"Treino ({nome_b})", color="#9467bd")
    ax.plot(df_b["epoch"], df_b["test_classification_errors"], label=f"Teste ({nome_b})", color="#c5b0d5", linestyle="--")
    ax.set_title("Comparativo de classificacoes incorretas por epoca", fontsize=15)
    ax.set_xlabel("Epoca", fontsize=12)
    ax.set_ylabel("Amostras classificadas incorretamente", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    salvar(fig, "comparativo_erros_classificacao.png")


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "individual"

    if modo == "individual":
        nome = sys.argv[2] if len(sys.argv) > 2 else "modelo"
        arquivos = csvs_mais_recentes(DIR_CSV, 1)
        if not arquivos:
            print(f"ERRO: nenhum arquivo .csv encontrado em '{DIR_CSV}'.")
            return 1
        print(f"Arquivo: {os.path.basename(arquivos[0])}")
        plotar_individual(arquivos[0], nome)

    elif modo == "comparativo":
        nome_a = sys.argv[2] if len(sys.argv) > 2 else "recente"
        nome_b = sys.argv[3] if len(sys.argv) > 3 else "anterior"
        arquivos = csvs_mais_recentes(DIR_CSV, 2)
        if not arquivos:
            print(f"ERRO: sao necessarios ao menos dois arquivos .csv em '{DIR_CSV}'.")
            return 1
        print(f"Mais recente ({nome_a}): {os.path.basename(arquivos[0])}")
        print(f"Anterior    ({nome_b}): {os.path.basename(arquivos[1])}")
        plotar_comparativo(arquivos[0], arquivos[1], nome_a, nome_b)

    else:
        print(f"ERRO: modo '{modo}' invalido. Use 'individual' ou 'comparativo'.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
