
/**
 * Fabrica de individuos aleatorios para a funcao Langermann.
 *
 * <p>Guarda o unico parametro fixo do problema — a dimensao — e o repassa ao construtor
 * de {@link LangermannInd} a cada chamada. E o que permite ao motor {@link Ag} montar a
 * populacao inicial sem conhecer o problema.</p>
 */
public class LangermannIndFactory implements IndFactory {

    /** dimensao do problema; deve ser 2. */
    private final int n;

    /**
     * Cria uma fabrica para uma instancia especifica do problema.
     *
     * @param n dimensao do problema; deve ser 2
     */
    public LangermannIndFactory(int n) {
        this.n = n;
    }

    /**
     * Cria um novo individuo aleatorio com a dimensao configurada nesta fabrica.
     *
     * @return individuo recem-sorteado, valido e independente dos anteriores
     */
    @Override
    public LangermannInd getInstance() {
        return new LangermannInd(this.n);
    }
}
