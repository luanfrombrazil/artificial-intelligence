
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Indivíduo do problema das N-Rainhas.
 *
 * <p>O problema: posicionar N rainhas num tabuleiro N×N sem que nenhuma ataque outra.
 * Rainhas atacam na horizontal, na vertical e nas duas diagonais.</p>
 *
 * <p><b>Codificação.</b> O cromossomo é um vetor de N inteiros em que
 * {@code genes[coluna] = linha}. Essa escolha já elimina por construção todos os
 * ataques verticais — cada coluna tem exatamente uma rainha. Se o vetor for além disso
 * uma <em>permutação</em> de {@code 0..N-1}, os ataques horizontais também desaparecem
 * de graça, e resta apenas verificar as diagonais.</p>
 *
 * <p><b>Correção histórica.</b> O construtor aleatório sorteava valores no intervalo
 * {@code 0..N} (inclusive), ou seja, N+1 linhas possíveis num tabuleiro de N linhas.
 * Isso permitia genes com o valor N — uma linha que não existe. Como a avaliação só
 * conta colisões entre rainhas, um tabuleiro com uma linha extra é mais fácil de
 * resolver, e o algoritmo declarava ótimas soluções impossíveis: em 100 execuções com
 * N=6, nenhuma solução era válida. O laço agora vai até {@code i < n}.</p>
 */
public class NRainhasInd implements Ind {

    /** {@code genes[coluna] = linha} da rainha daquela coluna. Valores em {@code 0..n-1}. */
    private final int[] genes;

    /** Ordem do tabuleiro: número de colunas, de linhas e de rainhas. */
    private final int n;

    /**
     * Cria um indivíduo aleatório: uma permutação das linhas {@code 0..n-1}.
     *
     * <p>Começar por uma permutação é uma vantagem da codificação — a população inicial
     * já nasce livre de ataques horizontais e verticais, e a evolução só precisa
     * trabalhar as diagonais.</p>
     *
     * @param n ordem do tabuleiro (número de rainhas); deve ser maior que 1
     */
    public NRainhasInd(int n) {
        List<Integer> numeros = new ArrayList<>();
        // As linhas válidas de um tabuleiro n x n são 0..n-1. Usar "i <= n" aqui
        // introduziria uma linha inexistente no espaço de busca.
        for (int i = 0; i < n; i++) {
            numeros.add(i);
        }

        Collections.shuffle(numeros, Rng.get());

        this.n = n;
        this.genes = new int[n];
        for (int i = 0; i < n; i++) {
            genes[i] = numeros.remove(0);
        }
    }

    /**
     * Construtor interno usado por {@link #recombinar(Ind)} e {@link #mutar()} para
     * materializar um indivíduo a partir de um vetor de genes já calculado.
     *
     * @param genes vetor de linhas por coluna; é copiado, de modo que alterações
     *              posteriores no array original não afetam o indivíduo
     */
    private NRainhasInd(int[] genes) {
        this.n = genes.length;
        this.genes = genes.clone();
    }

    /**
     * Recombina dois tabuleiros por crossover de um ponto.
     *
     * <p>Um ponto de corte é sorteado e os dois filhos herdam prefixo de um pai e
     * sufixo do outro, em ordens trocadas.</p>
     *
     * <p><b>Observação sobre o operador.</b> Corte de um ponto não preserva a propriedade
     * de permutação: os filhos podem repetir linhas. A avaliação absorve isso contando
     * repetição como colisão, então nada quebra — mas o espaço de busca deixa de ser
     * {@code n!} e passa a ser {@code n^n}. Operadores próprios para permutação (PMX,
     * Order Crossover, Cycle Crossover) preservariam a propriedade.</p>
     *
     * @param ind o outro progenitor; deve ser um {@code NRainhasInd} do mesmo tamanho
     * @return lista com exatamente dois filhos
     */
    @Override
    public List<Ind> recombinar(Ind ind) {
        NRainhasInd outro = (NRainhasInd) ind;

        int corte = Rng.get().nextInt(n - 1) + 1;

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

    /**
     * Muta o tabuleiro trocando pares de posições.
     *
     * <p>Cada posição tem 50% de chance de disparar uma troca com outra posição sorteada,
     * e ao menos uma troca é garantida. A troca preserva a propriedade de permutação
     * quando o indivíduo já é uma permutação.</p>
     *
     * <p><b>Observação sobre a intensidade.</b> Com probabilidade 0,5 por posição, a
     * mutação executa em média n/2 trocas — mais próximo de um reinício aleatório do que
     * de uma perturbação local. O valor usual na literatura é 1/n.</p>
     *
     * @return um novo tabuleiro mutado
     */
    @Override
    public Ind mutar() {
        int[] novosGenes = genes.clone();
        boolean mutou = false;

        for (int i = 0; i < n; i++) {
            if (Rng.get().nextDouble() < 0.5) {
                trocar(novosGenes);
                mutou = true;
            }
        }
        if (!mutou) {
            trocar(novosGenes);
        }

        return new NRainhasInd(novosGenes);
    }

    /**
     * Troca de lugar dois genes sorteados.
     *
     * @param g vetor de genes a ser modificado no lugar
     */
    private void trocar(int[] g) {
        int troca1 = Rng.get().nextInt(n);
        int troca2 = Rng.get().nextInt(n);
        int aux = g[troca1];
        g[troca1] = g[troca2];
        g[troca2] = aux;
    }

    /**
     * Conta o número de pares de rainhas que se atacam.
     *
     * <p>Para cada par de colunas verifica dois tipos de ataque: mesma linha
     * ({@code genes[i] == genes[j]}) e mesma diagonal (a diferença de linhas é igual à
     * diferença de colunas). Ataques verticais são impossíveis pela codificação.</p>
     *
     * <p>Este é um problema de minimização com ótimo conhecido e igual a zero: avaliação
     * 0 significa tabuleiro resolvido. É o único problema do projeto em que o critério de
     * parada por otimalidade de {@link Ag} funciona.</p>
     *
     * @return número de pares em conflito; 0 indica uma solução válida
     */
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

    /**
     * Verifica se este indivíduo é uma solução sintaticamente válida do problema:
     * todos os genes dentro do tabuleiro e sem repetição de linha.
     *
     * <p>Útil em testes de regressão — foi com esta verificação que o defeito do
     * intervalo {@code 0..n} foi confirmado.</p>
     *
     * @return {@code true} se os genes formam uma permutação de {@code 0..n-1}
     */
    public boolean isValido() {
        boolean[] visto = new boolean[n];
        for (int gene : genes) {
            if (gene < 0 || gene >= n || visto[gene]) {
                return false;
            }
            visto[gene] = true;
        }
        return true;
    }

    /**
     * Devolve uma cópia do cromossomo.
     *
     * @return novo vetor com as linhas ocupadas em cada coluna
     */
    public int[] getGenes() {
        return genes.clone();
    }

    /**
     * Representação textual usada nos logs de execução.
     *
     * @return string no formato {@code NRainhasInd{genes=[...], avaliacao=x}}
     */
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
