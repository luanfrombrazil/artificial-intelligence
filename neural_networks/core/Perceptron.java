package neural_networks.core;

import neural_networks.utils.Base;
import neural_networks.utils.Statistics;

import java.util.Random;

/**
 * Perceptron de camada única — a rede neural na sua forma original.
 *
 * <p>Cada neurônio de saída calcula uma soma ponderada das entradas mais um viés e passa
 * o resultado por uma sigmoide. Não há camada intermediária: a saída é função direta da
 * entrada.</p>
 *
 * <p><b>O que essa arquitetura consegue e o que não consegue.</b> Um perceptron traça um
 * hiperplano no espaço de entrada e classifica de um lado ou de outro. Isso resolve
 * qualquer problema <em>linearmente separável</em> — AND, OR, NAND — e nenhum problema
 * que não seja. O XOR é o contraexemplo canônico: não existe reta que separe
 * {(0,0),(1,1)} de {(0,1),(1,0)}, e nenhum ajuste de pesos aqui vai encontrá-la. Essa
 * limitação, demonstrada por Minsky e Papert em 1969, é exatamente a razão de existir de
 * {@link MultilayerPerceptron} e de {@link NeuralNetwork}.</p>
 *
 * <p>O viés é implementado pela convenção clássica da entrada fixa: o vetor de entrada
 * ganha uma posição extra com valor 1, e o peso correspondente é o viés. Sem ele o
 * hiperplano seria obrigado a passar pela origem.</p>
 */
public class Perceptron {

    /** {@code w[i][j]}: peso da entrada {@code i} para o neurônio de saída {@code j}. A última linha é o viés. */
    private final double[][] w;

    /** Taxa de aprendizado. */
    private final double ni;

    /** Base de dados com treino e teste. */
    private final Base base;

    /** Número de épocas de treinamento. */
    private final int epochs;

    /** Número de atributos de entrada. */
    private final int in;

    /** Número de neurônios de saída. */
    private final int out;

    /** Semente usada na inicialização, guardada para registro. */
    private final long seed;

    /** Coletor de métricas por época. */
    public Statistics statistics;

    /**
     * Constrói um perceptron usando a semente padrão da base.
     *
     * @param base         base já carregada
     * @param epochs       número de épocas; valores menores ou iguais a zero viram 1000
     * @param ni           taxa de aprendizado
     * @param random_range amplitude da inicialização uniforme dos pesos
     */
    public Perceptron(Base base, int epochs, double ni, double random_range) {
        this(base, epochs, ni, random_range, Base.SEMENTE_PADRAO);
    }

    /**
     * Constrói um perceptron com semente explícita.
     *
     * @param base         base já carregada
     * @param epochs       número de épocas; valores menores ou iguais a zero viram 1000
     * @param ni           taxa de aprendizado
     * @param random_range amplitude da inicialização uniforme dos pesos
     * @param seed         semente do gerador pseudoaleatório, para reprodutibilidade
     */
    public Perceptron(Base base, int epochs, double ni, double random_range, long seed) {
        this.statistics = new Statistics(base.getName());
        this.base = base;
        this.epochs = epochs > 0 ? epochs : 1000;
        this.ni = ni;
        this.in = base.getIn();
        this.out = base.getOut();
        this.seed = seed;
        this.w = new double[in + 1][out];

        Random random = new Random(seed);
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                w[i][j] = random.nextDouble() * random_range * 2 - random_range;
            }
        }
    }

    /**
     * Treina o perceptron por {@link #epochs} épocas em gradiente descendente estocástico.
     *
     * <p>Para cada amostra: propaga, mede o erro, calcula o delta
     * {@code (alvo - saida) * f'(saida)} e ajusta os pesos na direção que reduz o erro.
     * Ao fim de cada época, avalia o conjunto de teste e registra as métricas.</p>
     */
    public void train() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochTrainTotalError = 0.0;
            int epochTrainClassificationErrors = 0;

            for (int line = 0; line < base.length; line++) {
                double[] x = montarEntrada(this.base.getInput_data()[line]);
                double[] guess = propagar(x);

                if (amostraIncorreta(guess, base.getOutput_data()[line])) {
                    epochTrainClassificationErrors++;
                }
                epochTrainTotalError += calculateError(base.getOutput_data()[line], guess);

                for (int j = 0; j < out; j++) {
                    double delta = (base.getOutput_data()[line][j] - guess[j]) * guess[j] * (1 - guess[j]);
                    for (int i = 0; i < in + 1; i++) {
                        w[i][j] += ni * delta * x[i];
                    }
                }
            }

            double[] testResults = test();
            statistics.recordEpoch(epochTrainTotalError / base.length, epochTrainClassificationErrors,
                    testResults[0] / base.test_length, (int) testResults[1]);
        }
    }

    /**
     * Avalia o perceptron no conjunto de teste, sem alterar os pesos.
     *
     * @return vetor de dois elementos: {@code [0]} soma dos erros, {@code [1]} número de
     *         amostras classificadas incorretamente
     */
    public double[] test() {
        double totalTestError = 0.0;
        int classificationErrors = 0;

        for (int line = 0; line < base.test_length; line++) {
            double[] x = montarEntrada(this.base.getTest_input_data()[line]);
            double[] guess = propagar(x);

            if (amostraIncorreta(guess, base.getTest_output_data()[line])) {
                classificationErrors++;
            }
            totalTestError += calculateError(base.getTest_output_data()[line], guess);
        }

        return new double[]{totalTestError, classificationErrors};
    }

    /**
     * Monta o vetor de entrada acrescentando a entrada fixa 1 que implementa o viés.
     *
     * @param amostra atributos da amostra, com {@link #in} elementos
     * @return novo vetor com {@code in + 1} elementos, o último igual a 1
     */
    private double[] montarEntrada(double[] amostra) {
        double[] x = new double[in + 1];
        System.arraycopy(amostra, 0, x, 0, in);
        x[in] = 1;
        return x;
    }

    /**
     * Propaga uma entrada já preparada através da camada de saída.
     *
     * @param x vetor de entrada com o termo de viés incluído
     * @return vetor de saídas, uma por neurônio
     */
    private double[] propagar(double[] x) {
        double[] guess = new double[out];
        for (int j = 0; j < out; j++) {
            double u = 0;
            for (int i = 0; i < in + 1; i++) {
                u += x[i] * w[i][j];
            }
            guess[j] = sigmoid(u);
        }
        return guess;
    }

    /**
     * Decide se uma amostra foi classificada incorretamente, por vencedor-leva-tudo.
     *
     * @param guess    saída produzida
     * @param esperado saída esperada
     * @return {@code true} se o neurônio mais ativado não corresponde à classe esperada
     */
    private boolean amostraIncorreta(double[] guess, double[] esperado) {
        int melhorGuess = 0;
        int melhorEsperado = 0;
        for (int i = 1; i < guess.length; i++) {
            if (guess[i] > guess[melhorGuess]) {
                melhorGuess = i;
            }
            if (esperado[i] > esperado[melhorEsperado]) {
                melhorEsperado = i;
            }
        }
        return melhorGuess != melhorEsperado;
    }

    /**
     * Função de ativação sigmoide logística.
     *
     * @param u entrada líquida do neurônio
     * @return valor entre 0 e 1
     */
    public double sigmoid(double u) {
        return 1 / (1 + Math.exp(-u));
    }

    /**
     * Soma dos desvios absolutos entre a saída esperada e a produzida.
     *
     * @param esperado saída esperada
     * @param guess    saída produzida
     * @return soma de {@code |esperado - obtido|} sobre todos os neurônios
     */
    public double calculateError(double[] esperado, double[] guess) {
        double total = 0.0;
        for (int j = 0; j < this.out; j++) {
            total += Math.abs(esperado[j] - guess[j]);
        }
        return total;
    }

    /** @return a semente usada na inicialização dos pesos */
    public long getSeed() {
        return seed;
    }

    /** Escreve o relatório detalhado de treinamento em {@code data/}. */
    public void printLog() {
        statistics.writeDetailedLog(base.length, base.test_length);
    }

    /** Exporta o histórico por época em CSV. */
    public void generateCSV() {
        statistics.exportToCSV();
    }
}
