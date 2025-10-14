# Início do código para copiar

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# --- FUNÇÕES DE BUSCA DE ARQUIVOS ---

def encontrar_csv_mais_recente(diretorio_alvo):
    """Encontra o arquivo .csv mais recente em um diretório."""
    try:
        padrao_de_busca = os.path.join(diretorio_alvo, '*.csv')
        lista_de_csvs = glob.glob(padrao_de_busca)
        if not lista_de_csvs:
            return None
        return max(lista_de_csvs, key=os.path.getmtime)
    except Exception as e:
        print(f"Erro ao procurar arquivo: {e}")
        return None


def encontrar_dois_csvs_mais_recentes(diretorio_alvo):
    """Encontra os dois arquivos .csv mais recentes em um diretório."""
    try:
        padrao_de_busca = os.path.join(diretorio_alvo, '*.csv')
        lista_de_csvs = glob.glob(padrao_de_busca)
        if len(lista_de_csvs) < 2:
            return None, None
        lista_ordenada = sorted(lista_de_csvs, key=os.path.getmtime, reverse=True)
        return lista_ordenada[0], lista_ordenada[1]
    except Exception as e:
        print(f"Erro ao procurar arquivos: {e}")
        return None, None


# --- FUNÇÕES DE PLOTAGEM ---

def plotar_grafico_individual(caminho_csv, nome_modelo):
    """Gera gráficos individuais para um único arquivo CSV."""
    print(f"Gerando graficos individuais para o modelo: {nome_modelo}")
    try:
        os.makedirs("img", exist_ok=True)
        df = pd.read_csv(caminho_csv, sep=',')
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV '{caminho_csv}': {e}")
        return

    # Gráfico MSE
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(df['epoch'], df['train_mse'], label='MSE de Treino', color='blue')
    ax1.plot(df['epoch'], df['test_mse'], label='MSE de Teste', color='orange', linestyle='--')
    ax1.set_title(f'MSE por Época ({nome_modelo})', fontsize=16)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
    nome_arquivo_mse = f'../img/grafico_mse_{nome_modelo}.png'
    plt.savefig(nome_arquivo_mse)
    print(f"Grafico salvo como '{nome_arquivo_mse}'")
    plt.close(fig1)

    # Gráfico de Erros
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(df['epoch'], df['train_classification_errors'], label='Erros de Treino', color='green')
    ax2.plot(df['epoch'], df['test_classification_errors'], label='Erros de Teste', color='red', linestyle='--')
    ax2.set_title(f'Erros de Classificacao por Época ({nome_modelo})', fontsize=16)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Numero de Amostras Erradas', fontsize=12)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    nome_arquivo_erros = f'../img/grafico_erros_{nome_modelo}.png'
    plt.savefig(nome_arquivo_erros)
    print(f"Grafico salvo como '{nome_arquivo_erros}'")
    plt.close(fig2)


def plotar_graficos_comparativos(caminho_csv_recente, caminho_csv_anterior):
    """Gera gráficos comparativos entre dois arquivos CSV."""
    print("Gerando graficos comparativos...")
    try:
        os.makedirs("img", exist_ok=True)
        df_recente = pd.read_csv(caminho_csv_recente, sep=',')
        df_anterior = pd.read_csv(caminho_csv_anterior, sep=',')
    except Exception as e:
        print(f"Ocorreu um erro ao ler os arquivos CSV: {e}")
        return

    # Gráfico Comparativo de MSE
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(df_recente['epoch'], df_recente['train_mse'], label='MSE Treino (NeuralNetwork)', color='blue')
    ax1.plot(df_recente['epoch'], df_recente['test_mse'], label='MSE Teste (NeuralNetwork)', color='deepskyblue',
             linestyle='--')
    ax1.plot(df_anterior['epoch'], df_anterior['train_mse'], label='MSE Treino (MLP)', color='red')
    ax1.plot(df_anterior['epoch'], df_anterior['test_mse'], label='MSE Teste (MLP)', color='tomato', linestyle='--')
    ax1.set_title('Comparativo de Mean Squared Error (MSE) por Época', fontsize=16)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
    nome_arquivo_mse = '../img/comparativo_mse_por_epoca.png'
    plt.savefig(nome_arquivo_mse)
    print(f"Grafico comparativo de MSE salvo como '{nome_arquivo_mse}'")
    plt.close(fig1)

    # Gráfico Comparativo de Erros
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    ax2.plot(df_recente['epoch'], df_recente['train_classification_errors'], label='Erros Treino (NeuralNetwork)',
             color='green')
    ax2.plot(df_recente['epoch'], df_recente['test_classification_errors'], label='Erros Teste (NeuralNetwork)',
             color='lime', linestyle='--')
    ax2.plot(df_anterior['epoch'], df_anterior['train_classification_errors'], label='Erros Treino (MLP)',
             color='purple')
    ax2.plot(df_anterior['epoch'], df_anterior['test_classification_errors'], label='Erros Teste (MLP)', color='violet',
             linestyle='--')
    ax2.set_title('Comparativo de Erros de Classificacao por Época', fontsize=16)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Numero de Amostras Erradas', fontsize=12)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    nome_arquivo_erros = '../img/comparativo_erros_classificacao.png'
    plt.savefig(nome_arquivo_erros)
    print(f"Grafico comparativo de erros salvo como '{nome_arquivo_erros}'")
    plt.close(fig2)


# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    diretorio_dos_csvs = '../data'

    modo = 'comparativo'
    if len(sys.argv) > 1:
        modo = sys.argv[1]

    # --- LÓGICA DE CONTROLE ---
    if modo == 'comparativo':
        print(
            f"MODO COMPARATIVO: Procurando pelos dois arquivos CSV mais recentes em '{os.path.abspath(diretorio_dos_csvs)}'")
        csv_recente, csv_anterior = encontrar_dois_csvs_mais_recentes(diretorio_dos_csvs)
        if csv_recente and csv_anterior:
            print("Arquivos encontrados:")
            print(f"  - Mais recente (NeuralNetwork): {os.path.basename(csv_recente)}")
            print(f"  - 2º mais recente (MLP):      {os.path.basename(csv_anterior)}")
            plotar_graficos_comparativos(csv_recente, csv_anterior)
        else:
            print("ERRO: Nao foi possivel encontrar pelo menos dois arquivos .csv no diretorio.")

    elif modo == 'individual':
        print(f"MODO INDIVIDUAL: Procurando pelo arquivo CSV mais recente em '{os.path.abspath(diretorio_dos_csvs)}'")
        csv_recente = encontrar_csv_mais_recente(diretorio_dos_csvs)
        if csv_recente:
            print(f"Arquivo encontrado: {os.path.basename(csv_recente)}")
            plotar_grafico_individual(csv_recente, sys.argv[2])
        else:
            print("ERRO: Nenhum arquivo .csv foi encontrado no diretorio.")
    else:
        print(f"ERRO: Modo '{modo}' invalido.")
        print("Modos validos: 'comparativo' ou 'individual'.")
        print("Exemplo de uso: python plotCharts.py individual")