package neural_networks.utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;
import java.util.Scanner;

/**
 * Carrega e prepara uma base de dados para treinamento supervisionado.
 *
 * <p>Uma base é composta por três conjuntos: <b>treino</b> (usado para ajustar os pesos),
 * <b>validação</b> (usado para decidir quando parar e comparar configurações) e
 * <b>teste</b> (usado uma única vez, ao final, para estimar generalização). Os dois
 * primeiros vêm do mesmo arquivo de origem; o terceiro é carregado separadamente e nunca
 * participa do treino.</p>
 *
 * <h2>Formato dos arquivos</h2>
 *
 * <p>O arquivo de teste, em {@code bases/datasets/<nome>/<nome>.test}, começa com três
 * inteiros — número de amostras, número de atributos de entrada, número de neurônios de
 * saída — seguidos das amostras, cada uma com as entradas e a saída esperada já
 * codificada em um-de-N.</p>
 *
 * <p>O treino vem particionado por classe em {@code bases/datasets/<nome>/classes/classN},
 * cada arquivo começando com a contagem de linhas e trazendo, por linha, as entradas
 * seguidas do rótulo bruto.</p>
 *
 * <h2>Balanceamento por replicação</h2>
 *
 * <p>Ao carregar o treino, todas as classes são levadas ao tamanho da maior por
 * replicação de amostras. {@link #randomize(ArrayList)} deveria introduzir uma
 * perturbação nas cópias, mas devolve a amostra intacta — o resultado é um conjunto de
 * treino majoritariamente formado por duplicatas exatas, com classes raras representadas
 * por uma única amostra repetida centenas de vezes. O comportamento é preservado de
 * propósito: a base faz parte do enunciado acadêmico. Ele precisa ser levado em conta ao
 * interpretar qualquer métrica de treino.</p>
 *
 * <h2>Codificação dos rótulos</h2>
 *
 * <p><b>Atenção.</b> O alvo de treino é montado a partir do <em>número do arquivo</em> de
 * classe ({@code classN}), que é um posto ordenado e não o rótulo em si. O arquivo de
 * teste usa a codificação direta do rótulo. Quando algum valor de rótulo está ausente da
 * base — o que acontece no abalone —, as duas codificações se descolam e passam a apontar
 * para neurônios diferentes. O comportamento é preservado por fazer parte do enunciado,
 * mas invalida a comparação direta entre métricas de treino e de teste.</p>
 */
public final class Base {

    /** Semente usada por padrão quando nenhuma é informada. */
    public static final long SEMENTE_PADRAO = 42L;

    /** Localidade dos arquivos de dados, que usam vírgula como separador decimal. */
    private static final Locale LOCALE_DADOS = Locale.forLanguageTag("pt-BR");

    /** Número de atributos de entrada. */
    private int in;

    /** Número de neurônios de saída. */
    private int out;

    /** Sugestão de tamanho de camada oculta, pela média entre entrada e saída. */
    private int hidden;

    /** Número de amostras de treino. */
    public int length;

    /** Atributos de entrada do treino, uma linha por amostra. */
    private double[][] input_data;

    /** Saídas esperadas do treino, uma linha por amostra. */
    private double[][] output_data;

    /** Número de amostras de validação; zero se a validação não foi separada. */
    public int validation_length;

    /** Atributos de entrada da validação. */
    private double[][] validation_input_data;

    /** Saídas esperadas da validação. */
    private double[][] validation_output_data;

    /** Número de amostras de teste. */
    public int test_length;

    /** Atributos de entrada do teste. */
    private double[][] test_input_data;

    /** Saídas esperadas do teste. */
    private double[][] test_output_data;

    /** Nome da base, em maiúsculas, usado nos logs e nos nomes de arquivo de saída. */
    private String name;

    /** Gerador semeado, compartilhado por embaralhamento e replicação. */
    private final Random random;

    /** Semente efetivamente usada, guardada para registro. */
    private final long seed;

    /**
     * Carrega uma base usando a semente padrão.
     *
     * @param name    nome da base, correspondente à pasta em {@code bases/datasets/}
     * @param classes número de arquivos de classe a carregar do subdiretório {@code classes/}
     */
    public Base(String name, int classes) {
        this(name, classes, SEMENTE_PADRAO);
    }

    /**
     * Carrega uma base com semente explícita.
     *
     * <p>A semente controla o embaralhamento e a replicação de amostras. Fixá-la torna o
     * conjunto de treino — e portanto todo o experimento — reprodutível.</p>
     *
     * @param name    nome da base, correspondente à pasta em {@code bases/datasets/}
     * @param classes número de arquivos de classe a carregar do subdiretório {@code classes/}
     * @param seed    semente do gerador pseudoaleatório
     */
    public Base(String name, int classes, long seed) {
        this.seed = seed;
        this.random = new Random(seed);

        carregarTeste(name);
        this.name = name.toUpperCase();
        carregarTreino(name, classes);
    }

    /**
     * Lê o arquivo de teste e define as dimensões da base.
     *
     * @param name nome da base
     */
    private void carregarTeste(String name) {
        String testPath = "bases/datasets/" + name + "/" + name + ".test";
        try (Scanner ts = new Scanner(new File(testPath)).useLocale(LOCALE_DADOS)) {
            this.test_length = ts.nextInt();
            this.in = ts.nextInt();
            this.out = ts.nextInt();
            this.hidden = (in + out) / 2;

            this.test_input_data = new double[test_length][in];
            this.test_output_data = new double[test_length][out];

            for (int l = 0; l < test_length; l++) {
                for (int j = 0; j < in; j++) {
                    test_input_data[l][j] = ts.nextDouble();
                }
                for (int j = 0; j < out; j++) {
                    test_output_data[l][j] = ts.nextDouble();
                }
            }
        } catch (IOException e) {
            System.err.println("ERRO AO LER ARQUIVO DE TESTE: " + testPath);
            e.printStackTrace();
        }
    }

    /**
     * Lê os arquivos de classe do treino e monta os vetores de entrada e de alvo,
     * replicando amostras até que todas as classes tenham o tamanho da maior.
     *
     * @param name    nome da base
     * @param classes número de arquivos de classe a percorrer, de 1 a {@code classes}
     */
    private void carregarTreino(String name, int classes) {
        String classesPath = "bases/datasets/" + name + "/classes/" + "class";

        int maior_classe = 0;
        for (int i = 1; i <= classes; i++) {
            try (Scanner sc = new Scanner(new File(classesPath + i)).useLocale(LOCALE_DADOS)) {
                int tamanho = sc.nextInt();
                if (tamanho > maior_classe) {
                    maior_classe = tamanho;
                }
            } catch (IOException e) {
                System.err.println("ERRO AO LER ARQUIVO DE TREINO: " + classesPath + i);
                e.printStackTrace();
            }
        }

        ArrayList<double[]> input = new ArrayList<>();
        ArrayList<double[]> output = new ArrayList<>();
        for (int i = 1; i <= classes; i++) {
            ArrayList<double[]> linhas_input = new ArrayList<>();
            ArrayList<double[]> linhas_output = new ArrayList<>();
            String filename = classesPath + i;
            try (Scanner sc = new Scanner(new File(filename)).useLocale(LOCALE_DADOS)) {
                int classsize = sc.nextInt();
                for (int j = 0; j < classsize; j++) {
                    double[] linhaIn = new double[in];
                    double[] linhaOut = new double[out];

                    for (int k = 0; k < in; k++) {
                        linhaIn[k] = sc.nextDouble();
                    }
                    sc.nextDouble(); // consome o rotulo bruto ao final da linha

                    linhaOut[linhaOut.length - i] = 1;
                    linhas_output.add(linhaOut);
                    linhas_input.add(linhaIn);
                }

                while (linhas_input.size() < maior_classe) {
                    linhas_input.add(randomize(linhas_input));
                    linhas_output.add(duplicate(linhas_output));
                }
            } catch (IOException e) {
                System.err.println("ERRO AO LER ARQUIVO DE TREINO: " + classesPath + i);
                e.printStackTrace();
            }
            input.addAll(linhas_input);
            output.addAll(linhas_output);
        }
        this.input_data = input.toArray(new double[0][0]);
        this.output_data = output.toArray(new double[0][0]);
        this.length = input_data.length;
        this.validation_length = 0;
    }

    /**
     * Sorteia uma amostra de entrada da lista para replicação.
     *
     * <p>O nome anuncia uma perturbação aleatória que nunca foi implementada: a amostra é
     * devolvida intacta, e o balanceamento acaba produzindo duplicatas exatas. Mantido
     * como está por fazer parte do enunciado da disciplina.</p>
     *
     * @param lista amostras de entrada já lidas para a classe corrente; não pode estar vazia
     * @return referência a uma das amostras existentes, sem alteração
     */
    public double[] randomize(ArrayList<double[]> lista) {
        double[] sample = lista.get(this.random.nextInt(lista.size()));
        //RANDOMIZAR A SAMPLE AQUI
        return sample;
    }

    /**
     * Sorteia uma saída da lista para acompanhar a entrada replicada.
     *
     * <p>Note que o sorteio é independente do de {@link #randomize(ArrayList)}: entrada e
     * saída da cópia não vêm necessariamente da mesma amostra original. Isso só é
     * inofensivo porque todas as saídas de uma mesma classe são idênticas.</p>
     *
     * @param lista saídas já montadas para a classe corrente; não pode estar vazia
     * @return referência a uma das saídas existentes
     */
    public double[] duplicate(ArrayList<double[]> lista) {
        double[] sample = lista.get(this.random.nextInt(lista.size()));
        return sample;
    }

    /**
     * Separa uma fração do conjunto de treino para servir de conjunto de validação.
     *
     * <p>A separação acontece <b>em memória</b>: nenhum arquivo é lido ou escrito, e as
     * bases em disco permanecem intactas. As amostras retiradas saem do treino, de modo
     * que os dois conjuntos são disjuntos.</p>
     *
     * <p>Chame sempre <b>depois</b> de {@link #shuffle()} e <b>antes</b> de
     * {@link #normalize()}: embaralhar primeiro garante que a validação não fique
     * composta apenas pelas últimas classes carregadas, e normalizar depois garante que
     * os parâmetros de escala sejam estimados apenas com o que sobrou no treino.</p>
     *
     * @param fracao proporção do treino a reservar para validação, entre 0 e 1
     *               (0,2 reserva 20%)
     * @throws IllegalArgumentException se {@code fracao} estiver fora de (0, 1) ou se o
     *                                  conjunto resultante ficaria vazio
     */
    public void separarValidacao(double fracao) {
        if (fracao <= 0 || fracao >= 1) {
            throw new IllegalArgumentException("A fracao de validacao deve estar entre 0 e 1 (exclusivos). Recebido: " + fracao);
        }
        int nValidacao = (int) Math.round(length * fracao);
        if (nValidacao < 1 || nValidacao >= length) {
            throw new IllegalArgumentException(
                    "Fracao de " + fracao + " sobre " + length + " amostras produziria um conjunto invalido.");
        }

        int nTreino = length - nValidacao;

        this.validation_input_data = new double[nValidacao][];
        this.validation_output_data = new double[nValidacao][];
        for (int i = 0; i < nValidacao; i++) {
            validation_input_data[i] = input_data[nTreino + i];
            validation_output_data[i] = output_data[nTreino + i];
        }

        double[][] novoInput = new double[nTreino][];
        double[][] novoOutput = new double[nTreino][];
        for (int i = 0; i < nTreino; i++) {
            novoInput[i] = input_data[i];
            novoOutput[i] = output_data[i];
        }

        this.input_data = novoInput;
        this.output_data = novoOutput;
        this.length = nTreino;
        this.validation_length = nValidacao;
    }

    /**
     * Informa se a base possui conjunto de validação separado.
     *
     * @return {@code true} se {@link #separarValidacao(double)} já foi chamado
     */
    public boolean temValidacao() {
        return validation_length > 0;
    }

    /**
     * Embaralha o conjunto de treino, mantendo cada entrada pareada com a sua saída.
     *
     * <p>Usa o algoritmo de Fisher-Yates com o gerador semeado da base, de modo que a
     * ordem resultante é reprodutível. Importa por dois motivos: as amostras são
     * carregadas agrupadas por classe, e o gradiente descendente estocástico é sensível à
     * ordem de apresentação.</p>
     */
    public void shuffle() {
        if (input_data.length == 0) {
            return;
        }

        for (int i = length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            if (i == j) {
                continue;
            }
            double[] bufferIn = input_data[i];
            double[] bufferOut = output_data[i];
            input_data[i] = input_data[j];
            output_data[i] = output_data[j];
            input_data[j] = bufferIn;
            output_data[j] = bufferOut;
        }
    }

    /**
     * Normaliza os atributos de entrada para o intervalo [0, 1] por escalonamento
     * min-máx.
     *
     * <p>Os valores mínimo e máximo de cada atributo são estimados <b>somente no conjunto
     * de treino</b> e depois aplicados à validação e ao teste. É o procedimento correto:
     * calcular a escala usando dados de teste seria vazamento de informação, porque o
     * modelo passaria a se beneficiar de estatísticas de um conjunto que deveria ser
     * desconhecido.</p>
     *
     * <p>Como consequência, valores de validação ou teste fora da faixa vista no treino
     * caem fora de [0, 1]. Isso é esperado e não deve ser recortado.</p>
     *
     * <p>Atributos constantes no treino recebem zero, evitando divisão por zero.</p>
     */
    public void normalize() {
        if (in == 0 || length == 0) {
            return;
        }

        double[] minValues = new double[in];
        double[] maxValues = new double[in];

        for (int j = 0; j < in; j++) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < length; i++) {
                if (input_data[i][j] < min) {
                    min = input_data[i][j];
                }
                if (input_data[i][j] > max) {
                    max = input_data[i][j];
                }
            }
            minValues[j] = min;
            maxValues[j] = max;
        }

        aplicarEscala(input_data, length, minValues, maxValues);
        if (temValidacao()) {
            aplicarEscala(validation_input_data, validation_length, minValues, maxValues);
        }
        aplicarEscala(test_input_data, test_length, minValues, maxValues);
    }

    /**
     * Aplica a escala min-máx a um conjunto, usando parâmetros estimados no treino.
     *
     * @param dados     matriz de entradas a transformar, no lugar
     * @param n         número de linhas a percorrer
     * @param minValues mínimo de cada atributo, estimado no treino
     * @param maxValues máximo de cada atributo, estimado no treino
     */
    private void aplicarEscala(double[][] dados, int n, double[] minValues, double[] maxValues) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < in; j++) {
                double range = maxValues[j] - minValues[j];
                dados[i][j] = (range == 0) ? 0 : (dados[i][j] - minValues[j]) / range;
            }
        }
    }

    /** @return número de atributos de entrada */
    public int getIn() {
        return in;
    }

    /** @return número de neurônios de saída */
    public int getOut() {
        return out;
    }

    /**
     * Sugestão de tamanho para uma camada oculta, calculada como a média entre o número
     * de entradas e o de saídas.
     *
     * <p>É uma regra prática, não um resultado teórico — serve como ponto de partida
     * razoável quando não há tempo para busca de hiperparâmetros.</p>
     *
     * @return {@code (in + out) / 2}
     */
    public int getHidden() {
        return hidden;
    }

    /** @return nome da base em maiúsculas */
    public String getName() {
        return name;
    }

    /** @return semente usada no carregamento e no embaralhamento */
    public long getSeed() {
        return seed;
    }

    /** @return matriz de entradas de treino */
    public double[][] getInput_data() {
        return input_data;
    }

    /** @return matriz de saídas esperadas de treino */
    public double[][] getOutput_data() {
        return output_data;
    }

    /** @return matriz de entradas de validação; {@code null} se não houver validação */
    public double[][] getValidation_input_data() {
        return validation_input_data;
    }

    /** @return matriz de saídas esperadas de validação; {@code null} se não houver validação */
    public double[][] getValidation_output_data() {
        return validation_output_data;
    }

    /** @return matriz de entradas de teste */
    public double[][] getTest_input_data() {
        return test_input_data;
    }

    /** @return matriz de saídas esperadas de teste */
    public double[][] getTest_output_data() {
        return test_output_data;
    }

    /**
     * Resumo textual das dimensões da base, útil no cabeçalho de um log de execução.
     *
     * @return uma linha com nome, dimensões e tamanho de cada conjunto
     */
    @Override
    public String toString() {
        return String.format("Base %s | entradas=%d saidas=%d | treino=%d validacao=%d teste=%d | semente=%d",
                name, in, out, length, validation_length, test_length, seed);
    }
}
