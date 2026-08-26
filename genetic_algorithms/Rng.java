import java.util.Random;

/**
 * Fonte única de números pseudoaleatórios do algoritmo genético.
 *
 * <p>Um algoritmo genético é estocástico: duas execuções com os mesmos parâmetros
 * produzem resultados diferentes. Para que um experimento seja reprodutível — e um
 * resultado de relatório seja verificável por terceiros — todo sorteio precisa vir
 * de um único gerador cuja semente pode ser fixada.</p>
 *
 * <p>Antes desta classe, cada implementação de {@link Ind} criava o seu próprio
 * {@code new Random()}, semeado pelo relógio do sistema. Não havia como repetir
 * uma execução, nem como comparar duas configurações sob o mesmo conjunto de
 * sorteios.</p>
 *
 * <p>Uso típico, no início de {@code main}:</p>
 * <pre>{@code
 * Rng.setSeed(42L);                 // execução reprodutível
 * Rng.setSeedAleatoria();           // execução independente (padrão)
 * long s = Rng.getSeed();           // recuperar a semente usada, para registrar no log
 * }</pre>
 *
 * <p>Para uma bateria de N execuções independentes mas reprodutíveis em conjunto,
 * use uma semente derivada do número da execução: {@code Rng.setSeed(sementeBase + i)}.</p>
 */
public final class Rng {

    /** Gerador compartilhado por todas as classes do projeto. */
    private static Random instancia;

    /** Semente efetivamente usada para criar {@link #instancia}. */
    private static long semente;

    static {
        setSeedAleatoria();
    }

    /** Classe utilitária: não deve ser instanciada. */
    private Rng() {
    }

    /**
     * Fixa a semente do gerador e reinicia a sequência de números pseudoaleatórios.
     * Chamadas subsequentes a {@link #get()} devolvem o gerador já reiniciado, de modo
     * que toda a execução passa a ser determinística.
     *
     * @param novaSemente semente a ser usada; qualquer valor {@code long} é válido
     */
    public static void setSeed(long novaSemente) {
        semente = novaSemente;
        instancia = new Random(novaSemente);
    }

    /**
     * Sorteia uma semente a partir do relógio do sistema e a fixa. Use quando quiser
     * uma execução independente, mas ainda assim registrada: a semente sorteada pode ser
     * lida com {@link #getSeed()} e reaplicada depois para repetir exatamente a execução.
     *
     * @return a semente sorteada e agora em uso
     */
    public static long setSeedAleatoria() {
        long s = System.nanoTime();
        setSeed(s);
        return s;
    }

    /**
     * Devolve a semente atualmente em uso. Deve ser registrada junto de qualquer
     * resultado experimental — sem ela o resultado não é reproduzível.
     *
     * @return a semente que gerou a sequência atual
     */
    public static long getSeed() {
        return semente;
    }

    /**
     * Devolve o gerador compartilhado. Todas as classes do projeto devem obter seus
     * sorteios por aqui, em vez de instanciar {@code new Random()}.
     *
     * @return o gerador pseudoaleatório único do projeto
     */
    public static Random get() {
        return instancia;
    }
}
