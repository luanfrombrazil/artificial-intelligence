
/**
 * Fabrica de individuos aleatorios para a funcao Perm beta.
 *
 * <p>Guarda o unico parametro fixo do problema — a dimensao — e o repassa ao construtor
 * de {@link PermInd} a cada chamada. E o que permite ao motor {@link Ag} montar a
 * populacao inicial sem conhecer o problema.</p>
 */
public class PermIndFactory implements IndFactory {

    /** dimensao do problema (numero de variaveis). */
    private final int n;

    /**
     * Cria uma fabrica para uma instancia especifica do problema.
     *
     * @param n dimensao do problema (numero de variaveis)
     */
    public PermIndFactory(int n) {
        this.n = n;
    }

    /**
     * Cria um novo individuo aleatorio com a dimensao configurada nesta fabrica.
     *
     * @return individuo recem-sorteado, valido e independente dos anteriores
     */
    @Override
    public PermInd getInstance() {
        return new PermInd(this.n);
    }
}
