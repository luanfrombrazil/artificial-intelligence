package neural_networks;

import neural_networks.activations.ActivationFunction;
import neural_networks.activations.ReLU;
import neural_networks.activations.Sigmoid;
import neural_networks.core.Inicializacao;
import neural_networks.core.NeuralNetwork;
import neural_networks.utils.Base;
import neural_networks.utils.GradientCheck;

import java.util.ArrayList;

/**
 * Ponto de entrada do projeto de redes neurais.
 *
 * <p>Carrega uma base, prepara os conjuntos, monta a rede e treina. A ordem das chamadas
 * de preparação não é arbitrária:</p>
 *
 * <ol>
 *   <li>{@link Base#shuffle()} — as amostras são carregadas agrupadas por classe;
 *       embaralhar antes de qualquer divisão evita que a validação fique composta apenas
 *       pelas últimas classes.</li>
 *   <li>{@link Base#separarValidacao(double)} — divide o treino em memória, sem tocar em
 *       nenhum arquivo.</li>
 *   <li>{@link Base#normalize()} — os parâmetros de escala precisam ser estimados apenas
 *       com o que sobrou no treino, ou a validação vaza informação para o modelo.</li>
 * </ol>
 *
 * <p>Executar {@link GradientCheck} antes do treino é barato e responde a uma pergunta
 * que nenhuma curva de erro responde: a retropropagação está calculando o gradiente
 * certo?</p>
 */
public class Main {

    /**
     * Executa uma sessão de treinamento com os parâmetros definidos no corpo do método.
     *
     * @param args não utilizado; os parâmetros são definidos no código
     */
    public static void main(String[] args) {

        // --- Verificacao da implementacao --------------------------------------
        // Confere se a retropropagacao calcula o gradiente correto, comparando-o com a
        // derivada numerica. Roda em redes sinteticas pequenas e nao depende da base.
        System.out.println("Verificando a implementacao da retropropagacao...");
        GradientCheck.Resultado check = verificarGradiente();
        System.out.println("  " + check);
        if (!check.aprovado()) {
            System.out.println("  ATENCAO: o gradiente diverge do numerico. Treinar agora produz numeros sem sentido.");
        }
        System.out.println();

        // --- Reprodutibilidade -------------------------------------------------
        // A mesma semente reproduz exatamente a mesma execucao: mesmo embaralhamento,
        // mesma divisao de validacao e mesma inicializacao de pesos.
        long semente = 42L;

        // --- Preparacao da base ------------------------------------------------
        Base base = new Base("abalone", 28, semente);
        base.shuffle();
        base.separarValidacao(0.2);   // 20% do treino viram validacao, em memoria
        base.normalize();             // escala estimada apenas no treino restante
        System.out.println(base);
        System.out.println();

        // --- Arquitetura -------------------------------------------------------
        // 4 camadas de neuronios => 3 camadas de pesos => 3 funcoes de ativacao.
        int[] architecture = {
                base.getIn(),      // atributos de entrada
                base.getHidden(),  // primeira camada oculta
                base.getHidden(),  // segunda camada oculta
                base.getOut()      // um neuronio por classe
        };

        ArrayList<ActivationFunction> functions = new ArrayList<>();
        functions.add(new ReLU());     // entrada  -> oculta 1
        functions.add(new ReLU());     // oculta 1 -> oculta 2
        functions.add(new Sigmoid());  // oculta 2 -> saida

        // --- Hiperparametros ---------------------------------------------------
        int epochs = 10000;           // teto; a parada antecipada costuma interromper antes
        double learningRate = 0.05;   // valores altos com ReLU tendem a matar a camada
        double randomRange = 0.3;     // usado apenas pela inicializacao UNIFORME

        // He dimensiona os pesos pelo numero de entradas da camada e e' a escolha
        // adequada para ReLU. Com a inicializacao UNIFORME original, esta arquitetura
        // satura nas primeiras epocas e para de aprender.
        NeuralNetwork rna = new NeuralNetwork(base, architecture, functions,
                epochs, learningRate, randomRange, Inicializacao.HE, semente);

        // Interrompe apos 50 epocas sem melhora na validacao e restaura os pesos da
        // melhor epoca. A decisao usa apenas a validacao: o teste nunca participa.
        rna.setEarlyStopping(50, 1e-5);

        rna.train();
        rna.printLog();
        rna.generateCSV();

        double[] teste = rna.test();
        System.out.printf("Resultado final no teste: %d erros em %d amostras (%.2f%% de acuracia)%n",
                (int) teste[1], base.test_length,
                100.0 * (base.test_length - teste[1]) / base.test_length);
        System.out.println("Semente: " + semente + " | epocas executadas: " + rna.getEpocasExecutadas()
                + " | melhor epoca: " + rna.getMelhorEpoca());
    }

    /**
     * Roda uma verificação de gradiente numa rede pequena e sintética.
     *
     * <p>Usa sigmoide em todas as camadas de propósito: a ReLU não é diferenciável na
     * origem, e uma perturbação que atravesse esse ponto produz divergência legítima que
     * seria confundida com defeito.</p>
     *
     * @return o resultado da verificação
     */
    private static GradientCheck.Resultado verificarGradiente() {
        Base fake = null;
        int[] arq = {4, 6, 3};
        ArrayList<ActivationFunction> fs = new ArrayList<>();
        fs.add(new Sigmoid());
        fs.add(new Sigmoid());

        java.util.List<neural_networks.core.Layer> layers = new ArrayList<>();
        java.util.Random r = new java.util.Random(7L);
        for (int i = 0; i < arq.length - 1; i++) {
            layers.add(new neural_networks.core.Layer(arq[i], arq[i + 1], fs.get(i),
                    0.5, i == arq.length - 2, Inicializacao.XAVIER, r));
        }

        double[] entrada = {0.31, -0.72, 0.55, 0.08};
        double[] alvo = {0, 1, 0};
        return GradientCheck.verificar(layers, entrada, alvo,
                GradientCheck.EPSILON_PADRAO, GradientCheck.TOLERANCIA_PADRAO);
    }
}
