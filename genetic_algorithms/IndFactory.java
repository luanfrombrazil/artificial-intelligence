/**
 * Fábrica de indivíduos aleatórios de um problema.
 *
 * <p>O motor {@link Ag} precisa criar a população inicial sem saber qual problema está
 * resolvendo. Esta interface resolve isso: cada problema fornece uma fábrica que já
 * carrega os seus parâmetros fixos (a dimensão do vetor, o tamanho do tabuleiro) e sabe
 * produzir indivíduos válidos e aleatórios sob demanda.</p>
 *
 * <p>É o padrão <em>Factory Method</em> aplicado ao ponto exato onde o motor precisaria,
 * de outra forma, conhecer o construtor concreto de cada {@link Ind}.</p>
 */
public interface IndFactory {

    /**
     * Cria um novo indivíduo aleatório, válido para o problema desta fábrica.
     *
     * <p>Cada chamada deve produzir um indivíduo independente — é assim que o motor
     * monta uma população inicial diversa. Os sorteios devem vir de {@link Rng#get()}.</p>
     *
     * @return um indivíduo recém-criado, com genes sorteados dentro do domínio do problema
     */
    public Ind getInstance();
}
