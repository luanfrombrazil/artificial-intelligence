
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LangermannInd implements Ind {

    private static final double LIMITE_INFERIOR = 0.0;
    private static final double LIMITE_SUPERIOR = 10.0;
    private static final Random rand = new Random();

    private final double[] genes;

    public LangermannInd(int nDim) {
        this.genes = new double[nDim];
        for (int i = 0; i < nDim; i++) {
            this.genes[i] = LIMITE_INFERIOR + (LIMITE_SUPERIOR - LIMITE_INFERIOR) * rand.nextDouble();
        }
    }

    public LangermannInd(double[] genes) {
        this.genes = Arrays.copyOf(genes, genes.length);
    }

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
        filhos.add(new LangermannInd(filho1));
        filhos.add(new LangermannInd(filho2));
        filhos.add(new LangermannInd(filho3));
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
        return new LangermannInd(novo);
    }

    @Override
    public String toString() {
        return Arrays.toString(genes)
                + " -> f = " + getAvaliacao();
    }

    @Override
    public double getAvaliacao() {
        int m = 5;
        double[] c = {1.0, 2.0, 5.0, 9.0, 0.5};
        double[][] A = {
            {3, 5},
            {5, 2},
            {2, 1},
            {1, 4},
            {7, 9}
        };

        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            double s = 0.0;
            for (int j = 0; j < genes.length; j++) {
                s += Math.pow(genes[j] - A[i][j], 2);
            }
            sum += c[i] * Math.exp(-s / Math.PI) * Math.cos(Math.PI * s);
        }
        return -sum;
    }

}
