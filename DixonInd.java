
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class DixonInd implements Ind {

    private final double LIMITE_INFERIOR;
    private final double LIMITE_SUPERIOR;
    private static final Random rand = new Random();

    private final double[] genes;

    public DixonInd(int nDim) {
        this.genes = new double[nDim];
        this.LIMITE_INFERIOR = -nDim;
        this.LIMITE_SUPERIOR = nDim;
        for (int i = 0; i < nDim; i++) {
            this.genes[i] = LIMITE_INFERIOR + (LIMITE_SUPERIOR - LIMITE_INFERIOR) * rand.nextDouble();
        }

    }

    public DixonInd(double[] genes) {
        this.LIMITE_INFERIOR = -genes.length;
        this.LIMITE_SUPERIOR = genes.length;
        this.genes = Arrays.copyOf(genes, genes.length);
    }

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

        double blxAlpha = 0.5;
        double[] filho3 = new double[n];
        for (int i = 0; i < n; i++) {
            double cmin = Math.min(pai1.genes[i], pai2.genes[i]);
            double cmax = Math.max(pai1.genes[i], pai2.genes[i]);
            double I = cmax - cmin;
            double min = cmin - blxAlpha * I;
            double max = cmax + blxAlpha * I;
            filho3[i] = min + (max - min) * rand.nextDouble();
            filho3[i] = Math.max(LIMITE_INFERIOR, Math.min(LIMITE_SUPERIOR, filho3[i]));
        }

        List<Ind> filhos = new ArrayList<>();
        filhos.add(new DixonInd(filho1));
        filhos.add(new DixonInd(filho2));
        filhos.add(new DixonInd(filho3));
        return filhos;
    }

    @Override
    public Ind mutar() {
        double sigma = 0.1;
        double probMut = 0.2;
        double[] novo = Arrays.copyOf(genes, genes.length);
        for (int i = 0; i < novo.length; i++) {
            if (rand.nextDouble() < probMut) {
                novo[i] += rand.nextGaussian() * sigma;
                novo[i] = Math.max(LIMITE_INFERIOR, Math.min(LIMITE_SUPERIOR, novo[i]));
            }
        }
        return new DixonInd(novo);
    }

    @Override
    public double getAvaliacao() {
        double funcao = Math.pow(genes[0] - 1, 2);
        for (int i = 1; i < genes.length; i++) {
            double term = 2 * Math.pow(genes[i], 2) - genes[i - 1];
            funcao += (i + 1) * Math.pow(term, 2);
        }
        return funcao;
    }

    @Override
    public String toString() {
        return Arrays.toString(genes)
                + " -> f = " + getAvaliacao();
    }
}
