
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NRainhasInd implements Ind {

    private final int[] genes;
    private final int n;
    private static final Random random = new Random();

    public NRainhasInd(int n) {
        this.n = n;
        this.genes = new int[n];
        for (int i = 0; i < n; i++) {
            genes[i] = random.nextInt(n);
        }
    }

    //FACILITAR A CRIAÇÃO DE MUTANTES E FILHOS A PARTIR DO VETOR DE GENES
    private NRainhasInd(int[] genes) {
        this.n = genes.length;
        this.genes = genes.clone();
    }

    @Override
    public List<Ind> recombinar(Ind ind) {
        NRainhasInd outro = (NRainhasInd) ind;

        int corte = random.nextInt(n - 1) + 1;

        int[] filho1 = new int[n];
        int[] filho2 = new int[n];

        System.arraycopy(this.genes, 0, filho1, 0, corte);
        System.arraycopy(outro.genes, corte, filho1, corte, n - corte);

        System.arraycopy(outro.genes, 0, filho2, 0, corte);
        System.arraycopy(this.genes, corte, filho2, corte, n - corte);

        List<Ind> filhos = new ArrayList<>();
        filhos.add(new NRainhasInd(filho1));
        filhos.add(new NRainhasInd(filho2));
        return filhos;
    }

    @Override
    public Ind mutar() {
        int[] novosGenes = genes.clone();
        boolean mutou = false;

        for (int i = 0; i < n; i++) {
            if (random.nextDouble() < 0.2) {
                int novoValor;
                do {
                    novoValor = random.nextInt(n);
                } while (novoValor == novosGenes[i]);
                novosGenes[i] = novoValor;
                mutou = true;
            }
        }
        if (!mutou) {
            int pos = random.nextInt(n);
            int novoValor;
            do {
                novoValor = random.nextInt(n);
            } while (novoValor == novosGenes[pos]);
            novosGenes[pos] = novoValor;
        }

        return new NRainhasInd(novosGenes);
    }

    @Override
    public double getAvaliacao() {
        int colisoes = 0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (genes[i] == genes[j]) {
                    colisoes++;
                }
                if (Math.abs(genes[i] - genes[j]) == Math.abs(i - j)) {
                    colisoes++;
                }
            }
        }
        return colisoes;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NRainhasInd{genes=[");
        for (int i = 0; i < n; i++) {
            sb.append(genes[i]);
            if (i < n - 1) {
                sb.append(", ");
            }
        }
        sb.append("], avaliacao=").append(getAvaliacao()).append("}");
        return sb.toString();
    }
}
