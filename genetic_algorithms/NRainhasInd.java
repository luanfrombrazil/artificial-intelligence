
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class NRainhasInd implements Ind {

    private final int[] genes;
    private final int n;
    private static final Random random = new Random();

    public NRainhasInd(int n) {
        List<Integer> numeros = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            numeros.add(i);
        }

        Collections.shuffle(numeros);

        this.n = n;
        this.genes = new int[n];
        for (int i = 0; i < n; i++) {
            genes[i] = numeros.remove(0);
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
            if (random.nextDouble() < 0.5) {
                int troca1 = random.nextInt(n);
                int troca2 = random.nextInt(n);
                int aux = novosGenes[troca1];
                novosGenes[troca1] = novosGenes[troca2];
                novosGenes[troca2] = aux;
                mutou = true;
            }
        }
        if (!mutou) {
            int troca1 = random.nextInt(n);
            int troca2 = random.nextInt(n);
            int aux = novosGenes[troca1];
            novosGenes[troca1] = novosGenes[troca2];
            novosGenes[troca2] = aux;
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
