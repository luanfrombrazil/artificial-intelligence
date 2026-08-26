package neural_networks.core;

import neural_networks.activations.ActivationFunction;
import neural_networks.utils.Base;
import neural_networks.utils.Statistics;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Perceptron multicamadas genérico, com número arbitrário de camadas e função de ativação
 * escolhida por camada.
 *
 * <p>É a evolução de {@link MultilayerPerceptron}, que tinha uma única camada oculta e a
 * sigmoide fixa no código. Aqui a topologia é descrita por um vetor de inteiros — a
 * "arquitetura" — e cada camada de pesos recebe a sua própria
 * {@link ActivationFunction}:</p>
 *
 * <pre>{@code
 * int[] arquitetura = { 10, 19, 19, 29 };   // 10 entradas, duas ocultas de 19, 29 saidas
 * // 4 camadas de neuronios => 3 camadas de pesos => 3 funcoes de ativacao
 * }</pre>
 *
 * <p>O treinamento é gradiente descendente estocástico: cada amostra provoca um passo
 * completo de propagação, retropropagação e atualização de pesos.</p>
 *
 * <h2>Conjunto de validação e parada antecipada</h2>
 *
 * <p>Se {@link Base#temValidacao()} for verdadeiro, o erro de validação é medido a cada
 * época e usado para duas coisas: guardar o melhor estado da rede já visto e interromper
 * o treino quando esse estado deixa de melhorar por {@code paciencia} épocas seguidas.
 * Ao final, os pesos do melhor momento são restaurados.</p>
 *
 * <p>A separação importa metodologicamente. Escolher a época de parada olhando o conjunto
 * de teste faz o teste participar da construção do modelo, e ele deixa de estimar
 * generalização. O conjunto de teste continua sendo medido a cada época aqui, mas apenas
 * para desenhar a curva — nenhuma decisão do treino o consulta.</p>
 *
 * <h2>Reprodutibilidade</h2>
 *
 * <p>Toda a inicialização de pesos vem de um {@link Random} semeado pelo parâmetro
 * {@code seed}. Duas execuções com a mesma semente, a mesma base e os mesmos
 * hiperparâmetros produzem resultados idênticos.</p>
 */
public class NeuralNetwork {

    /** Camadas de pesos, da entrada para a saída. */
    private final List<Layer> layers = new ArrayList<>();

    /** Número máximo de épocas de treinamento. */
    private final int epochs;

    /** Taxa de aprendizado aplicada em todas as camadas. */
    private final double learningRate;

    /** Base de dados com os conjuntos de treino, validação e teste. */
    private final Base base;

    /** Semente usada na inicialização, guardada para registro no log. */
    private final long seed;

    /** Coletor de métricas por época. */
    public Statistics statistics;

    /** Épocas sem melhora na validação toleradas antes de interromper. Zero desliga. */
    private int paciencia = 0;

    /** Melhora mínima no erro de validação para que uma época conte como progresso. */
    private double minDelta = 1e-6;

    /** Época em que o melhor erro de validação foi observado (1-based); -1 se não houve. */
    private int melhorEpoca = -1;

    /** Melhor erro médio de validação observado. */
    private double melhorErroValidacao = Double.POSITIVE_INFINITY;

    /** Época em que o treino efetivamente parou (1-based). */
    private int epocasExecutadas = 0;

    /**
     * Constrói a rede com a configuração completa.
     *
     * @param base          base já carregada; fornece treino, validação (se houver) e teste
     * @param architecture  tamanhos das camadas de neurônios, da entrada para a saída.
     *                      Deve ter ao menos dois elementos; o primeiro precisa ser igual a
     *                      {@link Base#getIn()} e o último a {@link Base#getOut()}
     * @param functions     uma função de ativação por camada de pesos, ou seja
     *                      {@code architecture.length - 1} elementos
     * @param epochs        número máximo de épocas
     * @param learningRate  taxa de aprendizado
     * @param randomRange   amplitude da inicialização uniforme; ignorado se
     *                      {@code inicializacao} não for {@link Inicializacao#UNIFORME}
     * @param inicializacao estratégia de inicialização dos pesos
     * @param seed          semente do gerador pseudoaleatório, para reprodutibilidade
     * @throws IllegalArgumentException se a arquitetura tiver menos de duas camadas ou se
     *                                  o número de funções não corresponder ao número de
     *                                  camadas de pesos
     */
    public NeuralNetwork(Base base, int[] architecture, List<ActivationFunction> functions,
                         int epochs, double learningRate, double randomRange,
                         Inicializacao inicializacao, long seed) {

        if (architecture == null || architecture.length < 2) {
            throw new IllegalArgumentException("A arquitetura precisa de ao menos duas camadas (entrada e saida).");
        }
        if (functions == null || functions.size() != architecture.length - 1) {
            throw new IllegalArgumentException(
                    "Sao necessarias " + (architecture.length - 1) + " funcoes de ativacao para uma arquitetura de "
                    + architecture.length + " camadas. Recebidas: " + (functions == null ? 0 : functions.size()));
        }

        this.base = base;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.seed = seed;
        this.statistics = new Statistics(base.getName());

        Random random = new Random(seed);

        // Uma camada de pesos por par de camadas de neuronios consecutivas. A ultima
        // delas - e somente ela - e' a camada de saida.
        //
        // A versao anterior testava "i == architecture.length - 1" dentro de um laco que
        // termina em length-2, condicao que nunca e' satisfeita; todas as camadas
        // acabavam marcadas como camada de saida.
        for (int i = 0; i < architecture.length - 1; i++) {
            boolean ehCamadaDeSaida = (i == architecture.length - 2);
            layers.add(new Layer(architecture[i], architecture[i + 1], functions.get(i),
                    randomRange, ehCamadaDeSaida, inicializacao, random));
        }
    }

    /**
     * Constrói a rede com inicialização uniforme, preservando a assinatura histórica.
     *
     * @param base         base já carregada
     * @param architecture tamanhos das camadas de neurônios
     * @param functions    uma função de ativação por camada de pesos
     * @param epochs       número máximo de épocas
     * @param learningRate taxa de aprendizado
     * @param randomRange  amplitude da inicialização uniforme
     * @param seed         semente do gerador pseudoaleatório
     */
    public NeuralNetwork(Base base, int[] architecture, ArrayList<ActivationFunction> functions,
                         int epochs, double learningRate, double randomRange, long seed) {
        this(base, architecture, functions, epochs, learningRate, randomRange, Inicializacao.UNIFORME, seed);
    }

    /**
     * Configura a parada antecipada com base no erro de validação.
     *
     * <p>Só tem efeito se a base tiver conjunto de validação — ver
     * {@link Base#separarValidacao(double)}. Sem validação, o treino roda até a última
     * época e emite um aviso.</p>
     *
     * @param paciencia número de épocas consecutivas sem melhora toleradas antes de parar.
     *                  Zero ou negativo desliga a parada antecipada
     * @param minDelta  redução mínima no erro de validação para que a época conte como
     *                  melhora. Evita que oscilação numérica reinicie a contagem
     */
    public void setEarlyStopping(int paciencia, double minDelta) {
        this.paciencia = paciencia;
        this.minDelta = minDelta;
    }

    /**
     * Propaga uma amostra pela rede e devolve a saída da última camada.
     *
     * @param inputs vetor de atributos, de tamanho igual ao número de entradas da rede
     * @return vetor de saída, de tamanho igual ao número de neurônios da última camada
     */
    public double[] predict(double[] inputs) {
        double[] currentOutput = inputs;
        for (Layer layer : layers) {
            currentOutput = layer.forward(currentOutput);
        }
        return currentOutput;
    }

    /**
     * Treina a rede por até {@link #epochs} épocas, com parada antecipada se configurada.
     *
     * <p>A cada época: percorre todas as amostras de treino em gradiente descendente
     * estocástico, mede o desempenho em validação e em teste, registra tudo em
     * {@link #statistics} e decide se continua. Ao final, se houve validação, os pesos do
     * melhor momento são restaurados — o modelo devolvido é o da melhor época, não o da
     * última.</p>
     */
    public void train() {
        boolean usarEarlyStopping = paciencia > 0 && base.temValidacao();
        if (paciencia > 0 && !base.temValidacao()) {
            System.out.println("AVISO: parada antecipada pedida, mas a base nao tem conjunto de validacao. "
                    + "Chame base.separarValidacao(fracao) antes de treinar. O treino seguira ate a ultima epoca.");
        }

        List<double[]> melhoresParametros = null;
        int epocasSemMelhora = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochTrainTotalError = 0.0;
            int epochTrainClassificationErrors = 0;

            for (int i = 0; i < base.length; i++) {
                double[] input = base.getInput_data()[i];
                double[] target = base.getOutput_data()[i];

                double[] prediction = predict(input);

                double[] delta = new double[target.length];
                for (int j = 0; j < target.length; j++) {
                    delta[j] = target[j] - prediction[j];
                }

                for (int j = layers.size() - 1; j >= 0; j--) {
                    delta = layers.get(j).backward(delta);
                }

                for (Layer layer : layers) {
                    layer.updateParameters(learningRate);
                }

                epochTrainTotalError += calculateSampleError(target, prediction);
                if (wasMisclassified(target, prediction)) {
                    epochTrainClassificationErrors++;
                }
            }

            double[] validacao = base.temValidacao()
                    ? avaliarConjunto(base.getValidation_input_data(), base.getValidation_output_data(), base.validation_length)
                    : new double[]{0.0, 0.0};
            double[] teste = avaliarConjunto(base.getTest_input_data(), base.getTest_output_data(), base.test_length);

            double trainErro = epochTrainTotalError / base.length;
            double validacaoErro = base.temValidacao() ? validacao[0] / base.validation_length : Double.NaN;
            double testeErro = teste[0] / base.test_length;

            statistics.recordEpoch(trainErro, epochTrainClassificationErrors,
                    validacaoErro, (int) validacao[1],
                    testeErro, (int) teste[1]);

            epocasExecutadas = epoch + 1;

            if (base.temValidacao()) {
                if (validacaoErro < melhorErroValidacao - minDelta) {
                    melhorErroValidacao = validacaoErro;
                    melhorEpoca = epoch + 1;
                    epocasSemMelhora = 0;
                    melhoresParametros = exportarParametros();
                } else {
                    epocasSemMelhora++;
                }

                if (usarEarlyStopping && epocasSemMelhora >= paciencia) {
                    System.out.printf("%nPARADA ANTECIPADA na epoca %d: %d epocas sem melhora na validacao. "
                            + "Melhor epoca: %d (erro de validacao %.6f).%n",
                            epocasExecutadas, epocasSemMelhora, melhorEpoca, melhorErroValidacao);
                    break;
                }
            }
        }

        if (melhoresParametros != null) {
            importarParametros(melhoresParametros);
            statistics.setMelhorEpocaValidacao(melhorEpoca);
            System.out.printf("Pesos restaurados para os da epoca %d (melhor validacao).%n", melhorEpoca);
        }
        System.out.println();
    }

    /**
     * Mede erro médio e número de classificações incorretas em um conjunto qualquer,
     * sem alterar os pesos.
     *
     * @param entradas matriz de atributos, uma linha por amostra
     * @param alvos    matriz de saídas esperadas, uma linha por amostra
     * @param n        número de amostras a percorrer
     * @return vetor de dois elementos: {@code [0]} soma dos erros por amostra,
     *         {@code [1]} número de amostras classificadas incorretamente
     */
    private double[] avaliarConjunto(double[][] entradas, double[][] alvos, int n) {
        double totalErro = 0.0;
        int errosClassificacao = 0;

        for (int i = 0; i < n; i++) {
            double[] prediction = predict(entradas[i]);
            totalErro += calculateSampleError(alvos[i], prediction);
            if (wasMisclassified(alvos[i], prediction)) {
                errosClassificacao++;
            }
        }
        return new double[]{totalErro, errosClassificacao};
    }

    /**
     * Avalia a rede no conjunto de teste.
     *
     * @return vetor de dois elementos: {@code [0]} soma dos erros por amostra,
     *         {@code [1]} número de amostras classificadas incorretamente
     */
    public double[] test() {
        return avaliarConjunto(base.getTest_input_data(), base.getTest_output_data(), base.test_length);
    }

    /**
     * Avalia a rede no conjunto de validação.
     *
     * @return vetor de dois elementos, como em {@link #test()}; zeros se não houver
     *         conjunto de validação
     */
    public double[] validate() {
        if (!base.temValidacao()) {
            return new double[]{0.0, 0.0};
        }
        return avaliarConjunto(base.getValidation_input_data(), base.getValidation_output_data(), base.validation_length);
    }

    /**
     * Calcula o erro de uma amostra como soma dos desvios absolutos entre alvo e saída.
     *
     * <p><b>Nota sobre o nome.</b> Esta é a soma dos erros <em>absolutos</em>, não
     * quadráticos — é um SAE, não um MSE. O nome MSE aparecia no restante do projeto por
     * herança e não corresponde a esta conta.</p>
     *
     * @param target     saída esperada
     * @param prediction saída produzida pela rede
     * @return soma de {@code |alvo - saida|} sobre todos os neurônios de saída
     */
    private double calculateSampleError(double[] target, double[] prediction) {
        double error = 0;
        for (int i = 0; i < target.length; i++) {
            error += Math.abs(target[i] - prediction[i]);
        }
        return error;
    }

    /**
     * Decide se uma amostra foi classificada incorretamente, por vencedor-leva-tudo.
     *
     * <p>Compara o índice do maior valor da saída com o índice do maior valor do alvo.
     * Esta formulação substitui a anterior, que arredondava as saídas e comparava números
     * de ponto flutuante com {@code !=} — frágil por depender da representação binária
     * exata e por classificar como erro qualquer empate no valor máximo.</p>
     *
     * @param target     saída esperada, tipicamente codificada em um-de-N
     * @param prediction saída produzida pela rede
     * @return {@code true} se o neurônio mais ativado não corresponde à classe esperada
     */
    private boolean wasMisclassified(double[] target, double[] prediction) {
        return argmax(prediction) != argmax(target);
    }

    /**
     * Devolve o índice do maior valor de um vetor. Em caso de empate devolve o menor índice.
     *
     * @param v vetor não vazio
     * @return índice do valor máximo
     */
    private static int argmax(double[] v) {
        int melhor = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[melhor]) {
                melhor = i;
            }
        }
        return melhor;
    }

    /**
     * Copia os parâmetros de todas as camadas, para que possam ser restaurados depois.
     *
     * @return lista com um vetor achatado de parâmetros por camada, na ordem das camadas
     */
    private List<double[]> exportarParametros() {
        List<double[]> copia = new ArrayList<>(layers.size());
        for (Layer layer : layers) {
            copia.add(layer.exportarParametros());
        }
        return copia;
    }

    /**
     * Restaura os parâmetros de todas as camadas a partir de uma cópia.
     *
     * @param parametros lista produzida por {@link #exportarParametros()} sobre esta mesma rede
     */
    private void importarParametros(List<double[]> parametros) {
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).importarParametros(parametros.get(i));
        }
    }

    /**
     * Liga ou desliga a heurística {@code delta *= |1 - saida|} da camada de saída.
     *
     * <p>Ver {@link Layer#setHeuristicaAceleracao(boolean)}. Serve para reproduzir o
     * comportamento original do projeto; com ela ligada, a verificação numérica de
     * gradiente acusa divergência, porque o valor deixa de ser um gradiente.</p>
     *
     * @param ativa {@code true} para reativar a heurística
     */
    public void setHeuristicaAceleracao(boolean ativa) {
        for (Layer layer : layers) {
            layer.setHeuristicaAceleracao(ativa);
        }
    }

    /**
     * Devolve as camadas da rede, em ordem, para inspeção.
     *
     * <p>Usado por {@link neural_networks.utils.GradientCheck}, que precisa perturbar
     * pesos individualmente.</p>
     *
     * @return lista imodificável das camadas
     */
    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }

    /**
     * @return a semente usada para inicializar os pesos desta rede
     */
    public long getSeed() {
        return seed;
    }

    /**
     * @return a época de melhor erro de validação (1-based), ou -1 se não houve validação
     */
    public int getMelhorEpoca() {
        return melhorEpoca;
    }

    /**
     * @return quantas épocas o treino efetivamente executou antes de terminar ou parar
     */
    public int getEpocasExecutadas() {
        return epocasExecutadas;
    }

    /** Escreve o relatório detalhado de treinamento em {@code data/}. */
    public void printLog() {
        statistics.writeDetailedLog(base.length, base.validation_length, base.test_length);
    }

    /** Exporta o histórico por época em CSV, para geração de gráficos. */
    public void generateCSV() {
        statistics.exportToCSV();
    }
}
