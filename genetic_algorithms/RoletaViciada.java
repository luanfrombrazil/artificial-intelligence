
import java.util.ArrayList;
import java.util.List;

/**
 * Seleção por roleta viciada (<em>fitness proportionate selection</em>).
 *
 * <p>Cada indivíduo recebe uma fatia de uma roleta imaginária proporcional à sua
 * aptidão; sortear a roleta N vezes seleciona N indivíduos, com repetição permitida.
 * Como o projeto trabalha por padrão com minimização, a aptidão é obtida invertendo
 * a avaliação: quanto menor o valor da função objetivo, maior a fatia.</p>
 *
 * <p><b>Restrição importante — leia antes de usar em uma função nova.</b> A inversão
 * {@code 1/avaliacao} só é válida quando a avaliação é <em>estritamente positiva</em>:</p>
 * <ul>
 *   <li>Avaliação negativa produz aptidão negativa, e a soma acumulada deixa de ser
 *       monotônica. Na prática a seleção passa a ignorar os melhores indivíduos.
 *       É o caso da {@link LangermannInd}, cujo ótimo é negativo.</li>
 *   <li>Avaliação exatamente zero produz {@link Double#MAX_VALUE}, que domina a soma
 *       e zera a probabilidade de todos os demais.</li>
 * </ul>
 *
 * <p>Para funções que podem devolver valores negativos, a alternativa correta é
 * seleção por torneio (independe de escala e de sinal) ou escalonamento linear
 * {@code f' = (f_max + epsilon) - f}, que é sempre positivo e preserva a ordem.</p>
 */
public class RoletaViciada {

    /**
     * Seleciona indivíduos por roleta proporcional à aptidão.
     *
     * <p>A seleção é <b>com reposição</b>: o mesmo indivíduo pode ser escolhido mais de
     * uma vez, e é justamente assim que os melhores aumentam sua presença na próxima
     * geração.</p>
     *
     * @param populacao   candidatos à seleção; não deve estar vazia
     * @param nIndividuos quantidade de indivíduos a selecionar; pode ser maior que o
     *                    tamanho da população, já que a seleção é com reposição
     * @return lista com os indivíduos sorteados, na ordem em que foram sorteados.
     *         São as mesmas referências da população de entrada, não cópias
     */
    public List<Ind> roletar(List<Ind> populacao, int nIndividuos) {

        List<Double> inversos = new ArrayList<>();
        double somaInversos = 0.0;

        for (Ind ind : populacao) {
            double avaliacao = ind.getAvaliacao();
            double inverso = (avaliacao == 0) ? Double.MAX_VALUE : 1.0 / avaliacao;
            inversos.add(inverso);
            somaInversos += inverso;
        }

        List<Double> probabilidades = new ArrayList<>();
        double acumulado = 0.0;
        for (Double inverso : inversos) {
            acumulado += inverso / somaInversos;
            probabilidades.add(acumulado);
        }

        List<Ind> selecionados = new ArrayList<>(nIndividuos);
        for (int i = 0; i < nIndividuos; i++) {
            double r = Rng.get().nextDouble();
            for (int j = 0; j < probabilidades.size(); j++) {
                if (r <= probabilidades.get(j)) {
                    selecionados.add(populacao.get(j));
                    break;
                }
            }
        }
        return selecionados;
    }
}
