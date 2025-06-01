
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class PermInd implements Ind {

    private static final double LIMITE_INFERIOR = -10.0;
    private static final double LIMITE_SUPERIOR = 10.0;
    private static final Random rand = new Random();

    private final double[] genes;

    public PermInd(int nDim) {
        this.genes = new double[nDim];
        for (int i = 0; i < nDim; i++) {
            this.genes[i] = LIMITE_INFERIOR + (LIMITE_SUPERIOR - LIMITE_INFERIOR) * rand.nextDouble();
        }
    }

    public PermInd(double[] genes) {
        this.genes = Arrays.copyOf(genes, genes.length);
    }

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
        filhos.add(new PermInd(filho1));
        filhos.add(new PermInd(filho2));
        filhos.add(new PermInd(filho3));
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
        return new PermInd(novo);
    }

    @Override
    public double getAvaliacao() {
        final double beta = 10.0;
        double sum = 0.0;
        int n = genes.length;

        for (int i = 1; i <= n; i++) {
            double innerSum = 0.0;
            for (int j = 1; j <= n; j++) {
                double xj = genes[j - 1];
                double term = (j + beta) * (Math.pow(xj, i) - 1.0 / Math.pow(j, i));
                innerSum += term;
            }
            sum += Math.pow(innerSum, 2);
        }
        return sum;
    }

    @Override
    public String toString() {
        return //Arrays.toString(genes) + 
                " -> f = " + getAvaliacao();
    }
}
