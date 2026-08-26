
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Indivíduo para minimização da função <b>Perm beta</b>.
 *
 * <p>Definição, com beta = 10:</p>
 * <pre>
 *   f(x) = SOMA_{i=1}^{n} [ SOMA_{j=1}^{n} (j + beta) * (xj^i - 1/j^i) ]^2
 * </pre>
 *
 * <p>É uma função deliberadamente mal-condicionada: os termos crescem com potências
 * de ordem {@code i}, de modo que variáveis diferentes contribuem em escalas
 * radicalmente distintas. Testa a robustez do algoritmo a superfícies com vales muito
 * estreitos e gradientes desequilibrados.</p>
 *
 * <p><b>Ótimo global:</b> f = 0, em {@code xi = i} — ou seja, x = (1, 2, 3, ..., n).</p>
 *
 * <p><b>Cuidados numéricos.</b> O termo {@code xj^i} cresce como {@code |x|^n}. Com o
 * domínio {@code [-10, 10]} usado nesta implementação, valores de n moderados já saturam
 * o {@code double} e a avaliação vira {@code Infinity}. O domínio canônico da literatura
 * é {@code [-n, n]}, que reduz bastante o problema — a divergência entre os dois deve ser
 * declarada em qualquer relatório.</p>
 */
public class PermInd implements Ind {

    /** Valor da função no ótimo global, para uso como critério de parada e referência. */
    public static final double OTIMO_CONHECIDO = 0.0;

    /** Limite inferior do domínio de cada gene. */
    private static final double LIMITE_INFERIOR = -10.0;

    /** Limite superior do domínio de cada gene. */
    private static final double LIMITE_SUPERIOR = 10.0;

    /** Parâmetro beta da função, que controla o peso relativo dos termos. */
    private static final double BETA = 10.0;

    /** Vetor de variáveis reais que compõe a solução candidata. */
    private final double[] genes;

    /**
     * Cria um indivíduo aleatório com genes uniformemente distribuídos no domínio.
     *
     * @param nDim dimensão do problema (número de variáveis)
     */
    public PermInd(int nDim) {
        this.genes = new double[nDim];
        for (int i = 0; i < nDim; i++) {
            this.genes[i] = LIMITE_INFERIOR + (LIMITE_SUPERIOR - LIMITE_INFERIOR) * Rng.get().nextDouble();
        }
    }

    /**
     * Construtor usado pelos operadores genéticos para materializar um indivíduo a
     * partir de genes já calculados.
     *
     * @param genes vetor de variáveis; é copiado, não referenciado
     */
    public PermInd(double[] genes) {
        this.genes = Arrays.copyOf(genes, genes.length);
    }

    /**
     * Recombina dois vetores reais, produzindo três filhos — dois por crossover
     * aritmético e um por BLX-alfa.
     *
     * <p>Vale aqui a mesma observação de {@link DixonInd#recombinar(Ind)}: com
     * {@code alpha} fixo em 0,5 os dois primeiros filhos são idênticos entre si.</p>
     *
     * @param outro o outro progenitor; deve ser um {@code PermInd} de mesma dimensão
     * @return lista com três novos indivíduos
     */
    @Override
    public List<Ind> recombinar(Ind outro) {
        PermInd pai1 = this;
        PermInd pai2 = (PermInd) outro;
        int n = genes.length;

        double alpha = 0.5;
        double[] filho1 = new double[n];
        double[] filho2 = new double[n];
        for (int i = 0; i < n; i++) {
            filho1[i] = alpha * pai1.genes[i] + (1 - alpha) * pai2.genes[i];
            filho2[i] = alpha * pai2.genes[i] + (1 - alpha) * pai1.genes[i];
        }

        double[] filho3 = new double[n];
        for (int i = 0; i < n; i++) {
            double cmin = Math.min(pai1.genes[i], pai2.genes[i]);
            double cmax = Math.max(pai1.genes[i], pai2.genes[i]);
            double I = cmax - cmin;
            double min = cmin - alpha * I;
            double max = cmax + alpha * I;
            filho3[i] = min + (max - min) * Rng.get().nextDouble();
            filho3[i] = Math.max(LIMITE_INFERIOR, Math.min(LIMITE_SUPERIOR, filho3[i]));
        }

        List<Ind> filhos = new ArrayList<>();
        filhos.add(new PermInd(filho1));
        filhos.add(new PermInd(filho2));
        filhos.add(new PermInd(filho3));
        return filhos;
    }

    /**
     * Mutação gaussiana: cada gene tem 20% de chance de receber ruído normal de desvio
     * 0,1, recortado ao domínio.
     *
     * @return um novo indivíduo mutado
     */
    @Override
    public Ind mutar() {
        double sigma = 0.1;
        double probMut = 0.2;
        double[] novo = Arrays.copyOf(genes, genes.length);
        for (int i = 0; i < novo.length; i++) {
            if (Rng.get().nextDouble() < probMut) {
                novo[i] += Rng.get().nextGaussian() * sigma;
                novo[i] = Math.max(LIMITE_INFERIOR, Math.min(LIMITE_SUPERIOR, novo[i]));
            }
        }
        return new PermInd(novo);
    }

    /**
     * Avalia a função Perm beta no ponto representado por este indivíduo.
     *
     * @return o valor de f(x); sempre maior ou igual a zero. Pode devolver
     *         {@link Double#POSITIVE_INFINITY} para dimensões altas, por saturação das
     *         potências
     */
    @Override
    public double getAvaliacao() {
        double sum = 0.0;
        int n = genes.length;

        for (int i = 1; i <= n; i++) {
            double innerSum = 0.0;
            for (int j = 1; j <= n; j++) {
                double xj = genes[j - 1];
                double term = (j + BETA) * (Math.pow(xj, i) - 1.0 / Math.pow(j, i));
                innerSum += term;
            }
            sum += Math.pow(innerSum, 2);
        }
        return sum;
    }

    /**
     * Devolve uma cópia do vetor de variáveis.
     *
     * @return novo vetor com os valores dos genes
     */
    public double[] getGenes() {
        return genes.clone();
    }

    /**
     * @return representação textual com o vetor de genes e o valor da função
     */
    @Override
    public String toString() {
        return Arrays.toString(genes) + " -> f = " + getAvaliacao();
    }
}
