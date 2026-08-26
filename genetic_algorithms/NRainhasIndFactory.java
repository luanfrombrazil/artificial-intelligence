
/**
 * Fabrica de individuos aleatorios para a tabuleiro das N-Rainhas.
 *
 * <p>Guarda o unico parametro fixo do problema — a dimensao — e o repassa ao construtor
 * de {@link NRainhasInd} a cada chamada. E o que permite ao motor {@link Ag} montar a
 * populacao inicial sem conhecer o problema.</p>
 */
public class NRainhasIndFactory implements IndFactory {

    /** ordem do tabuleiro (numero de rainhas). */
    private final int n;

    /**
     * Cria uma fabrica para uma instancia especifica do problema.
     *
     * @param n ordem do tabuleiro (numero de rainhas)
     */
    public NRainhasIndFactory(int n) {
        this.n = n;
    }

    /**
     * Cria um novo individuo aleatorio com a dimensao configurada nesta fabrica.
     *
     * @return individuo recem-sorteado, valido e independente dos anteriores
     */
    @Override
    public NRainhasInd getInstance() {
        return new NRainhasInd(this.n);
    }
}
