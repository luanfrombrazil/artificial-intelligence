
import java.util.List;

/**
 * Contrato de um indivíduo do algoritmo genético.
 *
 * <p>Esta interface é o ponto de extensão do projeto: o motor evolutivo ({@link Ag})
 * não conhece nenhum problema específico, apenas indivíduos que sabem se recombinar,
 * se mutar e se avaliar. Trocar de problema — de N-Rainhas para Dixon-Price, por
 * exemplo — custa uma implementação desta interface e uma de {@link IndFactory},
 * sem tocar no motor.</p>
 *
 * <p>Convenções que toda implementação deve respeitar:</p>
 * <ul>
 *   <li><b>Imutabilidade.</b> {@link #recombinar(Ind)} e {@link #mutar()} devolvem
 *       indivíduos <em>novos</em> e nunca alteram {@code this} nem o argumento. O motor
 *       mantém pais e filhos vivos na mesma população.</li>
 *   <li><b>Validade.</b> Todo indivíduo devolvido por recombinação ou mutação deve ser
 *       uma solução sintaticamente válida do problema (uma permutação continua sendo
 *       permutação, um vetor real continua dentro do domínio).</li>
 *   <li><b>Aleatoriedade centralizada.</b> Sorteios devem vir de {@link Rng#get()},
 *       para que a execução seja reprodutível.</li>
 * </ul>
 */
public interface Ind {

    /**
     * Recombina este indivíduo com outro, produzindo a prole.
     *
     * <p>O número de filhos fica a critério da implementação: as classes de otimização
     * contínua devolvem três (dois de crossover aritmético e um BLX-alfa), a de
     * N-Rainhas devolve dois. O motor apenas acumula o que for devolvido.</p>
     *
     * @param ind o outro progenitor; deve ser da mesma implementação concreta que
     *            {@code this}, pois a recombinação faz cast para o tipo específico
     * @return lista não vazia de novos indivíduos; nem {@code this} nem {@code ind}
     *         são modificados
     */
    public List<Ind> recombinar(Ind ind);

    /**
     * Produz uma variação aleatória deste indivíduo.
     *
     * <p>É o operador responsável por manter diversidade e permitir escapar de ótimos
     * locais. A intensidade da perturbação é responsabilidade da implementação.</p>
     *
     * @return um novo indivíduo, mutado; {@code this} permanece inalterado
     */
    public Ind mutar();

    /**
     * Avalia este indivíduo segundo a função objetivo do problema.
     *
     * <p>Por convenção do projeto o motor <b>minimiza</b>: valores menores são melhores.
     * Problemas de maximização devem devolver o negativo da função, ou ser executados
     * com {@code isMin = false} em
     * {@link Ag#executar(int, int, IndFactory, int, boolean, boolean)}.</p>
     *
     * @return o valor da função objetivo para este indivíduo; pode ser negativo
     *         (a Langermann, por exemplo, tem ótimo negativo)
     */
    public double getAvaliacao();
}
