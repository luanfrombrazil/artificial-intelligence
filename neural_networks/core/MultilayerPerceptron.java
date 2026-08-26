package neural_networks.core;

import neural_networks.utils.Base;
import neural_networks.utils.Statistics;

import java.util.Random;

/**
 * Perceptron multicamadas com uma camada oculta e sigmoide fixa.
 *
 * <p>É a primeira implementação de retropropagação do projeto, escrita de forma direta:
 * duas matrizes de pesos ({@link #wh} da entrada para a oculta, {@link #w_theta} da
 * oculta para a saída), tudo em laços explícitos, sem abstração de camada. Foi mantida
 * ao lado de {@link NeuralNetwork} de propósito — a comparação entre as duas mostra o que
 * a generalização em {@link Layer} e {@link neural_networks.activations.ActivationFunction}
 * trouxe.</p>
 *
 * <p><b>O salto em relação ao {@link Perceptron}.</b> A camada oculta é o que permite
 * resolver problemas não linearmente separáveis. Os neurônios ocultos aprendem
 * representações intermediárias — no XOR, por exemplo, um neurônio pode aprender o OR e
 * outro o NAND, e a camada de saída combina os dois num AND, que é a decomposição
 * clássica do XOR. O que estava fora do alcance de um hiperplano passa a ser alcançável
 * pela composição de vários.</p>
 *
 * <p>O preço é que o erro deixa de ser diretamente atribuível a um peso: não existe alvo
 * conhecido para o neurônio oculto. A retropropagação resolve isso distribuindo o erro da
 * saída para trás, proporcionalmente aos pesos que o causaram — é o que os laços de
 * {@link #train()} fazem.</p>
 *
 * <p><b>Nota sobre a heurística.</b> O cálculo do delta da camada de saída inclui um fator
 * {@code |1 - saida|} que não pertence à retropropagação padrão. Ele foi preservado aqui
 * por fidelidade à implementação original da disciplina; em {@link NeuralNetwork} ele é
 * opcional e vem desligado por padrão. Ver
 * {@link Layer#setHeuristicaAceleracao(boolean)} e
 * {@link neural_networks.utils.GradientCheck}.</p>
 */
public class MultilayerPerceptron {

    /** Pesos da entrada para a camada oculta; a última linha é o viés. */
    private final double[][] wh;

    /** Pesos da camada oculta para a saída; a última linha é o viés. */
    private final double[][] w_theta;

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

    /** Número de neurônios na camada oculta. */
    private final int hidden;

    /** Semente usada na inicialização, guardada para registro. */
    private final long seed;

    /** Coletor de métricas por época. */
    public Statistics statistics;

    /**
     * Constrói a rede usando a semente padrão da base.
     *
     * @param base         base já carregada
     * @param epochs       número de épocas de treinamento
     * @param ni           taxa de aprendizado
     * @param random_range amplitude da inicialização uniforme dos pesos
     */
    public MultilayerPerceptron(Base base, int epochs, double ni, double random_range) {
        this(base, epochs, ni, random_range, Base.SEMENTE_PADRAO);
    }

    /**
     * Constrói a rede com semente explícita.
     *
     * @param base         base já carregada; a camada oculta usa {@link Base#getHidden()}
     * @param epochs       número de épocas de treinamento
     * @param ni           taxa de aprendizado
     * @param random_range amplitude da inicialização uniforme dos pesos
     * @param seed         semente do gerador pseudoaleatório, para reprodutibilidade
     */
    public MultilayerPerceptron(Base base, int epochs, double ni, double random_range, long seed) {
        this.base = base;
        this.ni = ni;
        this.hidden = base.getHidden();
        this.in = base.getIn();
        this.out = base.getOut();
        this.seed = seed;
        this.statistics = new Statistics(base.getName());
        this.epochs = epochs;

        Random random = new Random(seed);

        this.wh = new double[in + 1][hidden];
        this.w_theta = new double[hidden + 1][out];

        for (int i = 0; i < wh.length; i++) {
            for (int j = 0; j < wh[0].length; j++) {
                wh[i][j] = random.nextDouble() * random_range * 2 - random_range;
            }
        }
        for (int i = 0; i < w_theta.length; i++) {
            for (int j = 0; j < w_theta[0].length; j++) {
                w_theta[i][j] = random.nextDouble() * random_range * 2 - random_range;
            }
        }
    }

    /**
     * Treina a rede por {@link #epochs} épocas em gradiente descendente estocástico.
     *
     * <p>Cada amostra provoca um ciclo completo: propagação até a saída, cálculo dos
     * deltas da saída e da camada oculta, e atualização das duas matrizes de pesos.
     * A ordem importa — os deltas da camada oculta precisam ser calculados com os pesos
     * de saída <em>antes</em> de eles serem atualizados, porque a regra da cadeia usa os
     * valores que produziram o erro.</p>
     */
    public void train() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochTrainTotalError = 0.0;
            int epochTrainClassificationErrors = 0;

            for (int line = 0; line < base.length; line++) {
                double[] x = montarEntrada(this.base.getInput_data()[line]);
                double[] hidden_layer = propagarOculta(x);
                double[] guess = propagarSaida(hidden_layer);

                if (amostraIncorreta(guess, base.getOutput_data()[line])) {
                    epochTrainClassificationErrors++;
                }
                epochTrainTotalError += calculateError(base.getOutput_data()[line], guess);

                // Delta da camada de saida. O fator |1 - guess| e' a heuristica de
                // aceleracao herdada do enunciado, e nao faz parte da retropropagacao.
                double[] delta_theta = new double[out];
                for (int j = 0; j < out; j++) {
                    delta_theta[j] = (base.getOutput_data()[line][j] - guess[j])
                            * guess[j] * (1 - guess[j]) * Math.abs(1 - guess[j]);
                }

                // Delta da camada oculta: o erro da saida redistribuido para tras,
                // ponderado pelos pesos que o causaram.
                double[] delta_h = new double[hidden];
                for (int h = 0; h < hidden; h++) {
                    double soma = 0;
                    for (int j = 0; j < out; j++) {
                        soma += delta_theta[j] * w_theta[h][j];
                    }
                    delta_h[h] = soma * hidden_layer[h] * (1 - hidden_layer[h]);
                }

                for (int j = 0; j < out; j++) {
                    for (int h = 0; h < hidden + 1; h++) {
                        w_theta[h][j] += ni * delta_theta[j] * hidden_layer[h];
                    }
                }
                for (int h = 0; h < hidden; h++) {
                    for (int i = 0; i < in + 1; i++) {
                        wh[i][h] += ni * delta_h[h] * x[i];
                    }
                }
            }

            double[] testResults = test();
            statistics.recordEpoch(epochTrainTotalError / base.length, epochTrainClassificationErrors,
                    testResults[0] / base.test_length, (int) testResults[1]);
        }
    }

    /**
     * Avalia a rede no conjunto de teste, sem alterar os pesos.
     *
     * @return vetor de dois elementos: {@code [0]} soma dos erros, {@code [1]} número de
     *         amostras classificadas incorretamente
     */
    public double[] test() {
        double totalTestError = 0.0;
        int classificationErrors = 0;

        for (int line = 0; line < base.test_length; line++) {
            double[] x = montarEntrada(this.base.getTest_input_data()[line]);
            double[] hidden_layer = propagarOculta(x);
            double[] guess = propagarSaida(hidden_layer);

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
     * Calcula as ativações da camada oculta.
     *
     * @param x vetor de entrada com o termo de viés incluído
     * @return vetor com {@code hidden + 1} posições; a última é o viés fixo em 1, pronta
     *         para alimentar a camada de saída
     */
    private double[] propagarOculta(double[] x) {
        double[] hidden_layer = new double[hidden + 1];
        for (int h = 0; h < hidden; h++) {
            double u = 0;
            for (int i = 0; i < wh.length; i++) {
                u += x[i] * wh[i][h];
            }
            hidden_layer[h] = sigmoid(u);
        }
        hidden_layer[hidden] = 1;
        return hidden_layer;
    }

    /**
     * Calcula as ativações da camada de saída.
     *
     * @param hidden_layer ativações da camada oculta, com o termo de viés incluído
     * @return vetor de saídas, uma por neurônio de saída
     */
    private double[] propagarSaida(double[] hidden_layer) {
        double[] guess = new double[out];
        for (int j = 0; j < out; j++) {
            double u = 0;
            for (int h = 0; h < w_theta.length; h++) {
                u += hidden_layer[h] * w_theta[h][j];
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
