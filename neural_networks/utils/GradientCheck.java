package neural_networks.utils;

import neural_networks.activations.ActivationFunction;
import neural_networks.activations.ReLU;
import neural_networks.activations.Sigmoid;
import neural_networks.core.Inicializacao;
import neural_networks.core.Layer;
import neural_networks.core.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Verificação numérica do gradiente (<em>gradient checking</em>).
 *
 * <h2>Por que isto existe</h2>
 *
 * <p>A retropropagação é um algoritmo silencioso quando está errado. Um sinal trocado,
 * uma derivada aplicada à camada errada, um índice deslocado — nada disso lança exceção.
 * A rede simplesmente aprende menos, ou aprende devagar, ou não aprende, e o sintoma é
 * indistinguível de uma taxa de aprendizado mal escolhida ou de um problema difícil.
 * Foi exatamente assim que o defeito da marcação da camada de saída sobreviveu no
 * projeto: a rede rodava, produzia números, gerava gráficos — e treinava com um gradiente
 * que não era o gradiente.</p>
 *
 * <h2>A ideia</h2>
 *
 * <p>A derivada tem uma definição que não depende de nenhuma implementação: é o limite da
 * variação da função dividida pela variação do parâmetro. Aproximando esse limite pela
 * diferença centrada, para cada peso {@code w} individualmente:</p>
 *
 * <pre>
 *   dE/dw  aproximadamente  [ E(w + eps) - E(w - eps) ] / (2 * eps)
 * </pre>
 *
 * <p>Isso exige apenas duas propagações para a frente por peso — é lentíssimo, e por isso
 * inútil para treinar. Mas é uma referência independente: se o gradiente analítico
 * produzido pela retropropagação concorda com essa estimativa até várias casas decimais,
 * a implementação está correta. Se não concorda, há um defeito, e o índice do peso
 * divergente aponta a camada onde ele está.</p>
 *
 * <p>A diferença <em>centrada</em> — usando {@code w+eps} e {@code w-eps}, e não
 * {@code w} e {@code w+eps} — não é um detalhe: o erro de truncamento cai de ordem
 * {@code eps} para ordem {@code eps^2}, o que costuma valer duas ordens de grandeza a
 * mais de precisão.</p>
 *
 * <h2>Função de erro adotada</h2>
 *
 * <pre>
 *   E = 0,5 * SOMA_j (alvo_j - saida_j)^2
 * </pre>
 *
 * <p>É a função para a qual a formulação de {@link Layer} é o gradiente. Lembrando a
 * convenção de sinal daquela classe: o que ela guarda como "gradiente" é o
 * <em>negativo</em> de {@code dE/dw}, ou seja, já a direção de descida. A comparação aqui
 * leva isso em conta.</p>
 *
 * <h2>Como interpretar o resultado</h2>
 *
 * <p>A métrica é o erro relativo entre o gradiente analítico e o numérico. Com
 * {@code eps = 1e-5} e aritmética de precisão dupla:</p>
 *
 * <table border="1">
 *   <caption>Faixas típicas de erro relativo</caption>
 *   <tr><th>Erro relativo</th><th>Leitura</th></tr>
 *   <tr><td>menor que 1e-7</td><td>gradiente correto</td></tr>
 *   <tr><td>1e-7 a 1e-4</td><td>suspeito; aceitável apenas com ReLU, por causa do joelho em zero</td></tr>
 *   <tr><td>maior que 1e-4</td><td>há um defeito na retropropagação</td></tr>
 * </table>
 *
 * <p>Redes com ReLU merecem cuidado extra: perto de {@code u = 0} a função não é
 * diferenciável, e a perturbação pode atravessar o joelho, gerando divergência legítima
 * que não indica defeito algum. Verificar com sigmoide em todas as camadas isola essa
 * fonte de ruído.</p>
 *
 * <h2>Uso</h2>
 *
 * <pre>{@code
 * GradientCheck.Resultado r = GradientCheck.verificar(rede, entrada, alvo);
 * System.out.println(r);
 * if (!r.aprovado()) { ... }
 * }</pre>
 *
 * <p>Executar esta classe diretamente roda uma bateria de verificações sobre redes
 * pequenas e sintéticas, sem depender de nenhuma base de dados.</p>
 */
public final class GradientCheck {

    /** Perturbação padrão aplicada a cada parâmetro. */
    public static final double EPSILON_PADRAO = 1e-5;

    /** Erro relativo máximo tolerado por padrão. */
    public static final double TOLERANCIA_PADRAO = 1e-6;

    /** Classe utilitária: não deve ser instanciada. */
    private GradientCheck() {
    }

    /**
     * Resultado de uma verificação de gradiente.
     *
     * <p>Além do veredito, guarda onde ficou a maior divergência — é a informação que
     * transforma "algo está errado" em "está errado na camada 2".</p>
     */
    public static final class Resultado {

        /** Maior erro relativo encontrado entre todos os parâmetros verificados. */
        public final double maiorErroRelativo;

        /** Erro relativo médio sobre todos os parâmetros verificados. */
        public final double erroRelativoMedio;

        /** Quantidade de parâmetros efetivamente verificados. */
        public final int parametrosVerificados;

        /** Índice da camada onde ocorreu a maior divergência; -1 se nenhuma foi verificada. */
        public final int camadaPior;

        /** Descrição textual do parâmetro de maior divergência. */
        public final String parametroPior;

        /** Gradiente analítico no parâmetro de maior divergência. */
        public final double analiticoPior;

        /** Gradiente numérico no parâmetro de maior divergência. */
        public final double numericoPior;

        /** Tolerância usada para decidir a aprovação. */
        public final double tolerancia;

        /**
         * Cria um resultado de verificação.
         *
         * @param maiorErroRelativo     maior erro relativo observado
         * @param erroRelativoMedio     média dos erros relativos
         * @param parametrosVerificados quantidade de parâmetros percorridos
         * @param camadaPior            índice da camada de maior divergência
         * @param parametroPior         descrição do parâmetro de maior divergência
         * @param analiticoPior         valor analítico naquele parâmetro
         * @param numericoPior          valor numérico naquele parâmetro
         * @param tolerancia            limite usado para aprovar ou reprovar
         */
        Resultado(double maiorErroRelativo, double erroRelativoMedio, int parametrosVerificados,
                  int camadaPior, String parametroPior, double analiticoPior, double numericoPior,
                  double tolerancia) {
            this.maiorErroRelativo = maiorErroRelativo;
            this.erroRelativoMedio = erroRelativoMedio;
            this.parametrosVerificados = parametrosVerificados;
            this.camadaPior = camadaPior;
            this.parametroPior = parametroPior;
            this.analiticoPior = analiticoPior;
            this.numericoPior = numericoPior;
            this.tolerancia = tolerancia;
        }

        /**
         * Indica se a implementação passou na verificação.
         *
         * @return {@code true} se o maior erro relativo ficou dentro da tolerância
         */
        public boolean aprovado() {
            return maiorErroRelativo <= tolerancia;
        }

        /**
         * @return relatório em uma ou duas linhas, pronto para a saída padrão
         */
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("%s  maior erro relativo = %.3e  (media %.3e, %d parametros, tolerancia %.0e)",
                    aprovado() ? "[OK]     " : "[FALHOU] ",
                    maiorErroRelativo, erroRelativoMedio, parametrosVerificados, tolerancia));
            if (!aprovado()) {
                sb.append(String.format("%n           pior caso: camada %d, %s | analitico = %+.10f | numerico = %+.10f",
                        camadaPior, parametroPior, analiticoPior, numericoPior));
            }
            return sb.toString();
        }
    }

    /**
     * Verifica o gradiente de uma rede com os parâmetros padrão.
     *
     * @param rede    rede a verificar
     * @param entrada uma amostra de entrada
     * @param alvo    a saída esperada para essa amostra
     * @return o resultado da verificação
     */
    public static Resultado verificar(NeuralNetwork rede, double[] entrada, double[] alvo) {
        return verificar(rede.getLayers(), entrada, alvo, EPSILON_PADRAO, TOLERANCIA_PADRAO);
    }

    /**
     * Verifica o gradiente de uma rede, com controle sobre a perturbação e a tolerância.
     *
     * @param rede       rede a verificar
     * @param entrada    uma amostra de entrada
     * @param alvo       a saída esperada para essa amostra
     * @param epsilon    perturbação aplicada a cada parâmetro. Muito grande aumenta o erro
     *                   de truncamento; muito pequena, o de arredondamento. 1e-5 é um bom
     *                   compromisso em precisão dupla
     * @param tolerancia erro relativo máximo aceito para aprovar
     * @return o resultado da verificação
     */
    public static Resultado verificar(NeuralNetwork rede, double[] entrada, double[] alvo,
                                      double epsilon, double tolerancia) {
        return verificar(rede.getLayers(), entrada, alvo, epsilon, tolerancia);
    }

    /**
     * Verifica o gradiente de uma pilha de camadas.
     *
     * <p>O procedimento, para cada parâmetro:</p>
     * <ol>
     *   <li>propaga a amostra e retropropaga o erro, guardando o gradiente analítico;</li>
     *   <li>soma {@code epsilon} ao parâmetro e mede o erro;</li>
     *   <li>subtrai {@code epsilon} e mede o erro de novo;</li>
     *   <li>restaura o valor original e compara a diferença centrada com o analítico.</li>
     * </ol>
     *
     * <p>Os gradientes analíticos são copiados antes das perturbações, porque cada nova
     * propagação sobrescreve o estado interno das camadas.</p>
     *
     * @param layers     camadas da rede, da entrada para a saída
     * @param entrada    uma amostra de entrada
     * @param alvo       a saída esperada para essa amostra
     * @param epsilon    perturbação aplicada a cada parâmetro
     * @param tolerancia erro relativo máximo aceito para aprovar
     * @return o resultado da verificação
     */
    public static Resultado verificar(List<Layer> layers, double[] entrada, double[] alvo,
                                      double epsilon, double tolerancia) {

        // 1. Gradiente analitico, via retropropagacao.
        double[] saida = propagar(layers, entrada);
        double[] delta = new double[alvo.length];
        for (int j = 0; j < alvo.length; j++) {
            delta[j] = alvo[j] - saida[j];
        }
        for (int i = layers.size() - 1; i >= 0; i--) {
            delta = layers.get(i).backward(delta);
        }

        // Copiar antes de perturbar: cada propagacao seguinte sobrescreve o estado interno.
        List<double[][]> gradPesos = new ArrayList<>();
        List<double[]> gradVieses = new ArrayList<>();
        for (Layer layer : layers) {
            double[][] g = layer.getWeightGradients();
            double[][] copia = new double[g.length][];
            for (int i = 0; i < g.length; i++) {
                copia[i] = g[i].clone();
            }
            gradPesos.add(copia);
            gradVieses.add(layer.getBiasGradients().clone());
        }

        // 2. Gradiente numerico, parametro a parametro.
        double maiorErro = 0.0;
        double somaErros = 0.0;
        int contados = 0;
        int camadaPior = -1;
        String parametroPior = "";
        double analiticoPior = 0.0;
        double numericoPior = 0.0;

        for (int c = 0; c < layers.size(); c++) {
            Layer layer = layers.get(c);
            double[][] pesos = layer.getWeights();

            for (int i = 0; i < pesos.length; i++) {
                for (int j = 0; j < pesos[i].length; j++) {
                    double original = pesos[i][j];

                    pesos[i][j] = original + epsilon;
                    double eMais = erro(propagar(layers, entrada), alvo);

                    pesos[i][j] = original - epsilon;
                    double eMenos = erro(propagar(layers, entrada), alvo);

                    pesos[i][j] = original;

                    // Layer guarda a direcao de descida, ou seja -dE/dw. Invertemos o
                    // numerico para compara-los na mesma convencao.
                    double numerico = -(eMais - eMenos) / (2 * epsilon);
                    double analitico = gradPesos.get(c)[i][j];
                    double erroRel = erroRelativo(analitico, numerico);

                    somaErros += erroRel;
                    contados++;
                    if (erroRel > maiorErro) {
                        maiorErro = erroRel;
                        camadaPior = c;
                        parametroPior = "peso[" + i + "][" + j + "]";
                        analiticoPior = analitico;
                        numericoPior = numerico;
                    }
                }
            }

            double[] vieses = layer.getBiases();
            for (int j = 0; j < vieses.length; j++) {
                double original = vieses[j];

                vieses[j] = original + epsilon;
                double eMais = erro(propagar(layers, entrada), alvo);

                vieses[j] = original - epsilon;
                double eMenos = erro(propagar(layers, entrada), alvo);

                vieses[j] = original;

                double numerico = -(eMais - eMenos) / (2 * epsilon);
                double analitico = gradVieses.get(c)[j];
                double erroRel = erroRelativo(analitico, numerico);

                somaErros += erroRel;
                contados++;
                if (erroRel > maiorErro) {
                    maiorErro = erroRel;
                    camadaPior = c;
                    parametroPior = "vies[" + j + "]";
                    analiticoPior = analitico;
                    numericoPior = numerico;
                }
            }
        }

        return new Resultado(maiorErro, contados == 0 ? 0 : somaErros / contados, contados,
                camadaPior, parametroPior, analiticoPior, numericoPior, tolerancia);
    }

    /**
     * Propaga uma amostra por uma pilha de camadas.
     *
     * @param layers  camadas, da entrada para a saída
     * @param entrada vetor de entrada
     * @return vetor de saída da última camada
     */
    private static double[] propagar(List<Layer> layers, double[] entrada) {
        double[] saida = entrada;
        for (Layer layer : layers) {
            saida = layer.forward(saida);
        }
        return saida;
    }

    /**
     * Calcula o erro quadrático de uma amostra, {@code E = 0,5 * SOMA (alvo - saida)^2}.
     *
     * @param saida saída produzida pela rede
     * @param alvo  saída esperada
     * @return o valor do erro
     */
    private static double erro(double[] saida, double[] alvo) {
        double e = 0.0;
        for (int j = 0; j < alvo.length; j++) {
            double d = alvo[j] - saida[j];
            e += d * d;
        }
        return 0.5 * e;
    }

    /**
     * Erro relativo entre dois valores, normalizado pela soma das magnitudes.
     *
     * <p>A normalização evita tanto a divisão por zero quanto o falso alarme em
     * parâmetros cujo gradiente é legitimamente minúsculo — um desvio absoluto de 1e-12
     * num gradiente de 1e-11 não é um defeito.</p>
     *
     * @param a valor analítico
     * @param n valor numérico
     * @return o erro relativo, entre 0 e 2
     */
    private static double erroRelativo(double a, double n) {
        double denominador = Math.max(1e-12, Math.abs(a) + Math.abs(n));
        return Math.abs(a - n) / denominador;
    }

    /**
     * Executa uma bateria de verificações sobre redes pequenas e sintéticas.
     *
     * <p>Não depende de nenhum arquivo de dados: monta camadas diretamente, com entradas
     * e alvos fixos, e verifica arquiteturas de profundidades diferentes. O último caso
     * liga de propósito a heurística {@code delta *= |1 - saida|} e demonstra que ela
     * reprova na verificação — porque, uma vez aplicada, o valor calculado deixa de ser o
     * gradiente da função de erro.</p>
     *
     * @param args não utilizado
     */
    public static void main(String[] args) {
        System.out.println("=== VERIFICACAO NUMERICA DO GRADIENTE ===");
        System.out.println("E = 0,5 * SOMA (alvo - saida)^2   |   diferenca centrada, eps = " + EPSILON_PADRAO);
        System.out.println();

        boolean tudoOk = true;

        tudoOk &= caso("1 camada, sigmoide           (3 -> 2)",
                new int[]{3, 2}, sigmoides(1), false);

        tudoOk &= caso("2 camadas, sigmoide          (4 -> 5 -> 3)",
                new int[]{4, 5, 3}, sigmoides(2), false);

        tudoOk &= caso("3 camadas, sigmoide          (6 -> 8 -> 8 -> 4)",
                new int[]{6, 8, 8, 4}, sigmoides(3), false);

        tudoOk &= caso("4 camadas, sigmoide          (5 -> 7 -> 7 -> 7 -> 2)",
                new int[]{5, 7, 7, 7, 2}, sigmoides(4), false);

        List<ActivationFunction> mistas = new ArrayList<>();
        mistas.add(new ReLU());
        mistas.add(new ReLU());
        mistas.add(new Sigmoid());
        tudoOk &= caso("3 camadas, ReLU + sigmoide   (6 -> 9 -> 9 -> 4)",
                new int[]{6, 9, 9, 4}, mistas, false);

        System.out.println();
        System.out.println("Controle negativo - a mesma rede com a heuristica |1 - saida| ligada.");
        System.out.println("Espera-se REPROVACAO: com ela o valor calculado nao e' mais o gradiente de E.");
        boolean heuristicaReprovou = !caso("3 camadas, sigmoide + heuristica",
                new int[]{6, 8, 8, 4}, sigmoides(3), true);

        System.out.println();
        System.out.println("=========================================");
        System.out.println("Retropropagacao pura:      " + (tudoOk ? "APROVADA em todos os casos" : "FALHOU"));
        System.out.println("Controle negativo:         " + (heuristicaReprovou
                ? "reprovou como esperado" : "NAO reprovou - a heuristica pode estar desligada"));
        System.out.println("=========================================");

        if (!tudoOk || !heuristicaReprovou) {
            System.exit(1);
        }
    }

    /**
     * Monta uma rede sintética, verifica o seu gradiente e imprime o resultado.
     *
     * @param nome        rótulo do caso, exibido na saída
     * @param arquitetura tamanhos das camadas de neurônios
     * @param funcoes     uma função de ativação por camada de pesos
     * @param heuristica  se {@code true}, liga a heurística de aceleração na camada de saída
     * @return {@code true} se a verificação foi aprovada
     */
    private static boolean caso(String nome, int[] arquitetura, List<ActivationFunction> funcoes, boolean heuristica) {
        Random random = new Random(7L);
        List<Layer> layers = new ArrayList<>();
        for (int i = 0; i < arquitetura.length - 1; i++) {
            boolean saida = (i == arquitetura.length - 2);
            Layer layer = new Layer(arquitetura[i], arquitetura[i + 1], funcoes.get(i),
                    0.5, saida, Inicializacao.XAVIER, random);
            layer.setHeuristicaAceleracao(heuristica);
            layers.add(layer);
        }

        double[] entrada = new double[arquitetura[0]];
        for (int i = 0; i < entrada.length; i++) {
            entrada[i] = random.nextDouble() * 2 - 1;
        }
        double[] alvo = new double[arquitetura[arquitetura.length - 1]];
        alvo[random.nextInt(alvo.length)] = 1;

        Resultado r = verificar(layers, entrada, alvo, EPSILON_PADRAO, TOLERANCIA_PADRAO);
        System.out.printf("%-34s %s%n", nome, r);
        return r.aprovado();
    }

    /**
     * Cria uma lista com n instâncias de sigmoide.
     *
     * @param n quantidade de camadas de pesos
     * @return lista com n ativações sigmoides
     */
    private static List<ActivationFunction> sigmoides(int n) {
        List<ActivationFunction> l = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            l.add(new Sigmoid());
        }
        return l;
    }
}
