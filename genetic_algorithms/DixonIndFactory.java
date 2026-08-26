
/**
 * Fabrica de individuos aleatorios para a funcao Dixon-Price.
 *
 * <p>Guarda o unico parametro fixo do problema — a dimensao — e o repassa ao construtor
 * de {@link DixonInd} a cada chamada. E o que permite ao motor {@link Ag} montar a
 * populacao inicial sem conhecer o problema.</p>
 */
public class DixonIndFactory implements IndFactory {

    /** dimensao do problema (numero de variaveis). */
    private final int n;

    /**
     * Cria uma fabrica para uma instancia especifica do problema.
     *
     * @param n dimensao do problema (numero de variaveis)
     */
    public DixonIndFactory(int n) {
        this.n = n;
    }

    /**
     * Cria um novo individuo aleatorio com a dimensao configurada nesta fabrica.
     *
     * @return individuo recem-sorteado, valido e independente dos anteriores
     */
    @Override
    public DixonInd getInstance() {
        return new DixonInd(this.n);
    }
}
