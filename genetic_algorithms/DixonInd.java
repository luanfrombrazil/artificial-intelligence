
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Indivíduo para minimização da função <b>Dixon-Price</b>.
 *
 * <p>Definição:</p>
 * <pre>
 *   f(x) = (x1 - 1)^2 + SOMA_{i=2}^{n} i * (2*xi^2 - x_{i-1})^2
 * </pre>
 *
 * <p>É uma função unimodal em forma de vale longo, curvo e muito estreito. O desafio
 * não é escapar de ótimos locais, e sim seguir o fundo do vale: qualquer passo grande
 * demais sai dele, o que a torna um bom teste para a capacidade de refinamento fino do
 * algoritmo.</p>
 *
 * <p><b>Ótimo global:</b> f = 0, em {@code xi = 2^(-(2^i - 2) / 2^i)}.</p>
 *
 * <p><b>Domínio.</b> Esta implementação usa {@code [-n, n]}, que varia com a dimensão.
 * O domínio canônico da literatura é {@code [-10, 10]}, fixo. A diferença precisa ser
 * declarada em qualquer relatório que compare resultados com valores publicados.</p>
 */
public class DixonInd implements Ind {

    /** Valor da função no ótimo global, para uso como critério de parada e referência. */
    public static final double OTIMO_CONHECIDO = 0.0;

    /** Limite inferior do domínio de cada gene. */
    private final double LIMITE_INFERIOR;

    /** Limite superior do domínio de cada gene. */
    private final double LIMITE_SUPERIOR;

    /** Vetor de variáveis reais que compõe a solução candidata. */
    private final double[] genes;

    /**
     * Cria um indivíduo aleatório com genes uniformemente distribuídos no domínio.
     *
     * @param nDim dimensão do problema (número de variáveis); define também o domínio,
     *             que nesta implementação é {@code [-nDim, nDim]}
     */
    public DixonInd(int nDim) {
        this.genes = new double[nDim];
        this.LIMITE_INFERIOR = -nDim;
        this.LIMITE_SUPERIOR = nDim;
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
    public DixonInd(double[] genes) {
        this.LIMITE_INFERIOR = -genes.length;
        this.LIMITE_SUPERIOR = genes.length;
        this.genes = Arrays.copyOf(genes, genes.length);
    }

    /**
     * Recombina dois vetores reais, produzindo três filhos.
     *
     * <p>Os dois primeiros vêm de <b>crossover aritmético</b> (combinação convexa dos
     * pais) e o terceiro de <b>BLX-alfa</b>, que sorteia cada gene num intervalo
     * expandido em torno dos valores dos pais e portanto pode explorar fora do segmento
     * que os une.</p>
     *
     * <p><b>Observação sobre o operador.</b> Com {@code alpha} fixo em 0,5, os filhos 1 e
     * 2 são numericamente idênticos — {@code 0,5a + 0,5b} e {@code 0,5b + 0,5a} são o
     * mesmo número. Sortear {@code alpha} por gene tornaria os dois complementares, que
     * é a intenção do crossover aritmético.</p>
     *
     * @param outro o outro progenitor; deve ser um {@code DixonInd} de mesma dimensão
     * @return lista com três novos indivíduos
     */
    @Override
    public List<Ind> recombinar(Ind outro) {
        DixonInd pai1 = this;
        DixonInd pai2 = (DixonInd) outro;
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
        filhos.add(new DixonInd(filho1));
        filhos.add(new DixonInd(filho2));
        filhos.add(new DixonInd(filho3));
        return filhos;
    }

    /**
     * Mutação gaussiana: cada gene tem 20% de chance de receber um ruído normal de
     * desvio 0,1, com o resultado recortado ao domínio.
     *
     * <p>É uma perturbação local, apropriada para refinar uma solução já razoável —
     * complementa o BLX-alfa, que faz a exploração ampla.</p>
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
        return new DixonInd(novo);
    }

    /**
     * Avalia a função Dixon-Price no ponto representado por este indivíduo.
     *
     * @return o valor de f(x); sempre maior ou igual a zero, sendo zero o ótimo global
     */
    @Override
    public double getAvaliacao() {
        double funcao = Math.pow(genes[0] - 1, 2);
        for (int i = 1; i < genes.length; i++) {
            double term = 2 * Math.pow(genes[i], 2) - genes[i - 1];
            funcao += (i + 1) * Math.pow(term, 2);
        }
        return funcao;
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
