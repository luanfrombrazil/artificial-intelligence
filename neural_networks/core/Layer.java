package neural_networks.core;

import neural_networks.activations.ActivationFunction;

import java.util.Random;

/**
 * Uma camada densa (totalmente conectada) da rede neural.
 *
 * <p>A camada guarda a matriz de pesos que liga as suas entradas aos seus neurônios, um
 * viés por neurônio e a função de ativação aplicada a todos eles. Sabe fazer três coisas:
 * propagar um sinal para a frente ({@link #forward(double[])}), propagar o erro para trás
 * ({@link #backward(double[])}) e atualizar os próprios parâmetros
 * ({@link #updateParameters(double)}).</p>
 *
 * <p><b>Convenção de sinal dos gradientes.</b> O que esta classe chama de "gradiente" é o
 * <em>negativo</em> da derivada do erro em relação ao parâmetro — isto é, já é a direção
 * de descida. Por isso {@link #updateParameters(double)} soma
 * ({@code w += taxa * gradiente}) em vez de subtrair. A convenção vem de o delta ser
 * definido como {@code (alvo - saida)} e é mantida por coerência com a formulação clássica
 * do algoritmo, mas precisa ser lembrada ao comparar com o gradiente numérico.</p>
 *
 * <p><b>Estado entre chamadas.</b> {@link #forward(double[])} guarda entradas e saídas da
 * última amostra, porque {@link #backward(double[])} precisa delas. A consequência é que
 * a sequência forward → backward → update é obrigatória e não pode ser intercalada com
 * outra amostra.</p>
 */
public class Layer {

    /** {@code weights[i][j]}: peso da entrada {@code i} para o neurônio {@code j}. */
    private final double[][] weights;

    /** Um viés por neurônio. */
    private final double[] biases;

    /** Não-linearidade aplicada à entrada líquida de cada neurônio desta camada. */
    private final ActivationFunction activationFunction;

    /** Gradientes calculados no último {@link #backward(double[])}, na mesma forma de {@link #weights}. */
    private final double[][] weightGradients;

    /** Gradientes dos vieses calculados no último {@link #backward(double[])}. */
    private final double[] biasGradients;

    /** Entradas recebidas no último {@link #forward(double[])}. */
    private double[] lastInputs;

    /** Saídas produzidas no último {@link #forward(double[])}. */
    private double[] lastOutputs;

    /**
     * Indica se esta é a camada de saída da rede.
     *
     * <p>Historicamente este campo era atribuído por uma condição que nunca era
     * verdadeira, de modo que <em>todas</em> as camadas se consideravam camada de saída.
     * Hoje ele controla apenas uma coisa: a aplicação da heurística
     * {@link #heuristicaAceleracao}.</p>
     */
    private final boolean outputLayer;

    /**
     * Liga a heurística de aceleração {@code delta *= |1 - saida|} na camada de saída.
     *
     * <p>Esse fator aparecia no código original e não é o gradiente de nenhuma função de
     * erro padrão: ele multiplica o delta por algo próximo de zero justamente quando a
     * saída está próxima do alvo, desligando o aprendizado onde a rede está acertando.
     * Fica disponível como opção, para reproduzir o comportamento original, mas o padrão
     * é retropropagação pura. Com a heurística ligada,
     * {@link neural_networks.utils.GradientCheck} acusa divergência — corretamente, pois
     * o valor deixa de ser um gradiente.</p>
     */
    private boolean heuristicaAceleracao = false;

    /**
     * Cria uma camada densa com pesos e vieses sorteados.
     *
     * @param nInputs            número de entradas da camada (neurônios da camada anterior,
     *                           ou atributos, se for a primeira)
     * @param nNeurons           número de neurônios desta camada
     * @param activationFunction não-linearidade aplicada a todos os neurônios; não pode ser nula
     * @param randomRange        amplitude da inicialização uniforme, usada apenas quando
     *                           {@code inicializacao} é {@link Inicializacao#UNIFORME}:
     *                           os pesos são sorteados em {@code [-randomRange, +randomRange]}
     * @param outputLayer        {@code true} se esta é a última camada da rede
     * @param inicializacao      estratégia de inicialização dos pesos
     * @param random             gerador a ser usado nos sorteios; recebê-lo de fora é o que
     *                           torna a inicialização reprodutível a partir de uma semente
     */
    public Layer(int nInputs, int nNeurons, ActivationFunction activationFunction,
                 double randomRange, boolean outputLayer, Inicializacao inicializacao, Random random) {

        this.activationFunction = activationFunction;
        this.weights = new double[nInputs][nNeurons];
        this.biases = new double[nNeurons];
        this.outputLayer = outputLayer;

        for (int i = 0; i < nInputs; i++) {
            for (int j = 0; j < nNeurons; j++) {
                weights[i][j] = sortearPeso(inicializacao, randomRange, nInputs, random);
            }
        }
        for (int j = 0; j < nNeurons; j++) {
            // O viés é sempre inicializado pequeno e uniforme: ele não recebe sinal de
            // entrada, então não há fan-in para dimensionar.
            biases[j] = random.nextDouble() * randomRange * 2 - randomRange;
        }

        this.weightGradients = new double[nInputs][nNeurons];
        this.biasGradients = new double[nNeurons];
    }

    /**
     * Sorteia um peso segundo a estratégia de inicialização escolhida.
     *
     * @param inicializacao estratégia a aplicar
     * @param randomRange   amplitude usada apenas por {@link Inicializacao#UNIFORME}
     * @param nInputs       fan-in da camada, usado por Xavier e He
     * @param random        gerador de números pseudoaleatórios
     * @return um valor inicial para um peso
     */
    private static double sortearPeso(Inicializacao inicializacao, double randomRange, int nInputs, Random random) {
        switch (inicializacao) {
            case HE:
                return random.nextGaussian() * Math.sqrt(2.0 / nInputs);
            case XAVIER:
                return random.nextGaussian() * Math.sqrt(1.0 / nInputs);
            case UNIFORME:
            default:
                return random.nextDouble() * randomRange * 2 - randomRange;
        }
    }

    /**
     * Propaga um vetor de entrada através da camada.
     *
     * <p>Para cada neurônio calcula a entrada líquida — soma ponderada das entradas mais
     * o viés — e aplica a função de ativação. Entradas e saídas ficam guardadas para uso
     * do passo para trás.</p>
     *
     * @param inputs vetor de entrada, de tamanho igual ao {@code nInputs} da camada
     * @return vetor de saídas, de tamanho igual ao número de neurônios
     */
    public double[] forward(double[] inputs) {
        this.lastInputs = inputs;

        int nNeurons = biases.length;
        double[] outputs = new double[nNeurons];

        for (int j = 0; j < nNeurons; j++) {
            double u = 0;
            for (int i = 0; i < inputs.length; i++) {
                u += inputs[i] * weights[i][j];
            }
            u += biases[j];
            outputs[j] = this.activationFunction.forward(u);
        }

        this.lastOutputs = outputs;
        return outputs;
    }

    /**
     * Propaga o erro para trás, calculando os gradientes desta camada e o sinal de erro
     * a ser entregue à camada anterior.
     *
     * <p>É a regra da cadeia aplicada camada a camada, que é toda a ideia da
     * retropropagação: o erro que chega do lado da saída é multiplicado pela derivada da
     * ativação local, produzindo o delta do neurônio; o delta multiplica a entrada para
     * dar o gradiente do peso, e multiplica o peso para dar o erro que a camada anterior
     * deve receber.</p>
     *
     * <p>Este método apenas <em>calcula</em> os gradientes; os pesos só mudam em
     * {@link #updateParameters(double)}. A separação é essencial: o sinal devolvido à
     * camada anterior precisa ser calculado com os pesos antigos.</p>
     *
     * @param deltaFromNextLayer sinal de erro vindo da camada seguinte, com um valor por
     *                           neurônio desta camada. Na camada de saída é simplesmente
     *                           {@code (alvo - saida)}
     * @return sinal de erro a ser passado à camada anterior, com um valor por entrada
     *         desta camada
     */
    public double[] backward(double[] deltaFromNextLayer) {
        int nInputs = lastInputs.length;
        int nNeurons = biases.length;
        double[] deltaForPreviousLayer = new double[nInputs];

        for (int j = 0; j < nNeurons; j++) {
            double derivative = this.activationFunction.derivative(this.lastOutputs[j]);

            double delta = deltaFromNextLayer[j] * derivative;

            if (outputLayer && heuristicaAceleracao) {
                delta *= Math.abs(1 - this.lastOutputs[j]);
            }

            biasGradients[j] = delta;
            for (int i = 0; i < nInputs; i++) {
                weightGradients[i][j] = lastInputs[i] * delta;
                deltaForPreviousLayer[i] += weights[i][j] * delta;
            }
        }

        return deltaForPreviousLayer;
    }

    /**
     * Aplica os gradientes calculados no último {@link #backward(double[])}.
     *
     * <p>Soma em vez de subtrair porque, pela convenção de sinal desta classe, o
     * gradiente armazenado já aponta na direção de descida.</p>
     *
     * @param learningRate taxa de aprendizado; controla o tamanho do passo. Valores altos
     *                     aceleram o início do treino mas podem fazer a rede divergir ou
     *                     matar neurônios ReLU
     */
    public void updateParameters(double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] += learningRate * weightGradients[i][j];
            }
        }
        for (int j = 0; j < biases.length; j++) {
            biases[j] += learningRate * biasGradients[j];
        }
    }

    /**
     * Liga ou desliga a heurística {@code delta *= |1 - saida|} na camada de saída.
     *
     * @param ativa {@code true} para reproduzir o comportamento original do projeto,
     *              {@code false} (padrão) para retropropagação pura
     */
    public void setHeuristicaAceleracao(boolean ativa) {
        this.heuristicaAceleracao = ativa;
    }

    /**
     * @return {@code true} se esta camada é a camada de saída da rede
     */
    public boolean isOutputLayer() {
        return outputLayer;
    }

    /**
     * Devolve a matriz de pesos viva, não uma cópia.
     *
     * <p>Exposta para que {@link neural_networks.utils.GradientCheck} possa perturbar um
     * peso de cada vez e medir o efeito no erro. Fora desse uso, prefira não alterá-la
     * diretamente.</p>
     *
     * @return referência à matriz interna {@code [nInputs][nNeurons]}
     */
    public double[][] getWeights() {
        return weights;
    }

    /**
     * Devolve o vetor de vieses vivo, não uma cópia. Ver {@link #getWeights()}.
     *
     * @return referência ao vetor interno, com um viés por neurônio
     */
    public double[] getBiases() {
        return biases;
    }

    /**
     * @return referência à matriz de gradientes dos pesos calculada no último
     *         {@link #backward(double[])}
     */
    public double[][] getWeightGradients() {
        return weightGradients;
    }

    /**
     * @return referência ao vetor de gradientes dos vieses calculado no último
     *         {@link #backward(double[])}
     */
    public double[] getBiasGradients() {
        return biasGradients;
    }

    /**
     * Copia pesos e vieses para um vetor achatado.
     *
     * <p>Usado pela parada antecipada, que precisa guardar o melhor estado visto e
     * restaurá-lo depois que o treino terminar.</p>
     *
     * @return novo vetor contendo todos os pesos, em ordem, seguidos de todos os vieses
     */
    public double[] exportarParametros() {
        double[] copia = new double[weights.length * weights[0].length + biases.length];
        int k = 0;
        for (double[] linha : weights) {
            for (double w : linha) {
                copia[k++] = w;
            }
        }
        for (double b : biases) {
            copia[k++] = b;
        }
        return copia;
    }

    /**
     * Restaura pesos e vieses a partir de um vetor produzido por
     * {@link #exportarParametros()}.
     *
     * @param parametros vetor achatado com o mesmo formato e tamanho gerado pela
     *                   exportação desta mesma camada
     * @throws IllegalArgumentException se o tamanho do vetor não corresponder ao da camada
     */
    public void importarParametros(double[] parametros) {
        int esperado = weights.length * weights[0].length + biases.length;
        if (parametros.length != esperado) {
            throw new IllegalArgumentException(
                    "Tamanho incompativel ao restaurar parametros: esperado " + esperado
                    + ", recebido " + parametros.length);
        }
        int k = 0;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] = parametros[k++];
            }
        }
        for (int j = 0; j < biases.length; j++) {
            biases[j] = parametros[k++];
        }
    }
}
