
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Indivíduo para minimização da função <b>Langermann</b>.
 *
 * <p>Definição, com m = 5 termos:</p>
 * <pre>
 *   f(x) = - SOMA_{i=1}^{m} ci * exp(-si / PI) * cos(PI * si)
 *   onde  si = SOMA_j (xj - A[i][j])^2
 * </pre>
 *
 * <p>É o oposto da Dixon-Price em dificuldade: uma superfície fortemente multimodal,
 * cheia de picos e vales irregulares distribuídos sem simetria. Serve para testar a
 * capacidade de <em>exploração</em> do algoritmo — a de não ficar preso no primeiro
 * vale que encontrar.</p>
 *
 * <p><b>Atenção ao ótimo de referência.</b> Esta implementação, no domínio
 * {@code [0, 10]^2} e com os coeficientes abaixo, tem mínimo global em
 * <b>-9,25625</b>, em x aproximadamente (1,183; 4,053) — valor confirmado por busca
 * exaustiva em grade. O número -4,1558 que aparece com frequência em livros e sites
 * corresponde a outra parametrização da mesma família de funções. Comparar um resultado
 * desta implementação com aquele valor faz um resultado correto parecer errado.</p>
 *
 * <p><b>Aviso sobre a seleção.</b> Como a avaliação é negativa na região de interesse,
 * a {@link RoletaViciada} do projeto não funciona corretamente com esta função — ver a
 * documentação daquela classe.</p>
 *
 * <p><b>Limitação de dimensão.</b> A matriz {@code A} é fixa em 5x2, portanto apenas
 * {@code nDim = 2} é suportado. O construtor valida esse contrato.</p>
 */
public class LangermannInd implements Ind {

    /**
     * Mínimo global desta implementação em {@code [0, 10]^2}, obtido por busca em grade
     * densa. Difere do valor citado na maioria das referências, que usam outra
     * parametrização.
     */
    public static final double OTIMO_CONHECIDO = -9.25625;

    /** Limite inferior do domínio de cada gene. */
    private static final double LIMITE_INFERIOR = 0.0;

    /** Limite superior do domínio de cada gene. */
    private static final double LIMITE_SUPERIOR = 10.0;

    /** Número de termos do somatório. */
    private static final int M = 5;

    /** Pesos de cada termo. */
    private static final double[] C = {1.0, 2.0, 5.0, 9.0, 0.5};

    /** Centros dos termos: 5 pontos no plano. Fixa a função em duas dimensões. */
    private static final double[][] A = {
        {3, 5},
        {5, 2},
        {2, 1},
        {1, 4},
        {7, 9}
    };

    /** Vetor de variáveis reais que compõe a solução candidata. */
    private final double[] genes;

    /**
     * Cria um indivíduo aleatório com genes uniformemente distribuídos em
     * {@code [0, 10]}.
     *
     * @param nDim dimensão do problema; deve ser exatamente 2, pois a matriz de centros
     *             tem duas colunas
     * @throws IllegalArgumentException se {@code nDim} for diferente de 2
     */
    public LangermannInd(int nDim) {
        if (nDim != A[0].length) {
            throw new IllegalArgumentException(
                    "LangermannInd suporta apenas nDim = " + A[0].length
                    + " (a matriz de centros A e' " + M + "x" + A[0].length + "). Recebido: " + nDim);
        }
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
    public LangermannInd(double[] genes) {
        this.genes = Arrays.copyOf(genes, genes.length);
    }

    /**
     * Recombina dois vetores reais, produzindo três filhos — dois por crossover
     * aritmético e um por BLX-alfa.
     *
     * <p>Vale aqui a mesma observação de {@link DixonInd#recombinar(Ind)}: com
     * {@code alpha} fixo em 0,5 os dois primeiros filhos são idênticos entre si.</p>
     *
     * @param outro o outro progenitor; deve ser um {@code LangermannInd}
     * @return lista com três novos indivíduos
     */
    @Override
    public List<Ind> recombinar(Ind outro) {
        LangermannInd pai1 = this;
        LangermannInd pai2 = (LangermannInd) outro;
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
        filhos.add(new LangermannInd(filho1));
        filhos.add(new LangermannInd(filho2));
        filhos.add(new LangermannInd(filho3));
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
        return new LangermannInd(novo);
    }

    /**
     * Avalia a função Langermann no ponto representado por este indivíduo.
     *
     * @return o valor de f(x); <b>pode ser negativo</b>, e é justamente na região
     *         negativa que estão as boas soluções
     */
    @Override
    public double getAvaliacao() {
        double sum = 0.0;
        for (int i = 0; i < M; i++) {
            double s = 0.0;
            for (int j = 0; j < genes.length; j++) {
                s += Math.pow(genes[j] - A[i][j], 2);
            }
            sum += C[i] * Math.exp(-s / Math.PI) * Math.cos(Math.PI * s);
        }
        return -sum;
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
