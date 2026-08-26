
import java.util.ArrayList;
import java.util.List;

/**
 * Motor evolutivo do projeto — a implementação do algoritmo genético propriamente dito.
 *
 * <p>A classe é agnóstica ao problema: ela só conhece {@link Ind} e {@link IndFactory}.
 * O ciclo de uma geração é:</p>
 *
 * <ol>
 *   <li><b>Recombinação.</b> A população é percorrida aos pares e cada casal gera prole
 *       via {@link Ind#recombinar(Ind)}.</li>
 *   <li><b>Mutação.</b> Cada indivíduo da população gera um mutante via {@link Ind#mutar()}.</li>
 *   <li><b>União.</b> Pais, mutantes e filhos são reunidos num único conjunto candidato —
 *       é um modelo de substituição (mu + lambda), em que os pais competem com a prole.</li>
 *   <li><b>Elitismo.</b> Os {@code nElite} melhores candidatos passam diretamente para a
 *       próxima geração.</li>
 *   <li><b>Seleção.</b> O restante da população é preenchido por
 *       {@link RoletaViciada#roletar(List, int)} sobre os candidatos remanescentes.</li>
 * </ol>
 *
 * <p><b>Reprodutibilidade.</b> Toda a aleatoriedade do processo vem de {@link Rng}.
 * Fixe {@link Rng#setSeed(long)} antes de chamar {@link #executar} e a execução inteira
 * passa a ser determinística.</p>
 *
 * <p><b>Limitações conhecidas</b>, documentadas para que não sejam confundidas com
 * decisões de projeto:</p>
 * <ul>
 *   <li>Não há seleção de pais: os casais são formados por posição na lista
 *       ({@code i}, {@code i+1}), sem torneio nem roleta.</li>
 *   <li>Não há taxa de crossover nem de mutação: todo indivíduo cruza e todo indivíduo
 *       gera um mutante.</li>
 *   <li>O critério de parada por otimalidade compara a avaliação com zero, o que só faz
 *       sentido em problemas cujo ótimo é exatamente zero (N-Rainhas). Funções contínuas
 *       com ótimo diferente de zero sempre esgotam o número de gerações.</li>
 * </ul>
 */
public class Ag {

    /** Classe utilitária: todo o comportamento está em {@link #executar}. */
    private Ag() {
    }

    /**
     * Executa o algoritmo genético e devolve o melhor indivíduo encontrado.
     *
     * @param nPop            tamanho da população mantida entre gerações; deve ser maior
     *                        que {@code nElite}
     * @param nElite          quantidade de indivíduos que passam por elitismo direto a cada
     *                        geração, sem passar pela seleção. Valores altos em relação a
     *                        {@code nPop} aumentam a pressão seletiva e favorecem
     *                        convergência prematura
     * @param indFac          fábrica usada para criar a população inicial; determina qual
     *                        problema será resolvido
     * @param nGer            número máximo de gerações; funciona como orçamento de execução
     * @param isMin           {@code true} para minimizar a função objetivo (menor é melhor),
     *                        {@code false} para maximizar
     * @param imprimir_todos  quando {@code true}, imprime a população completa a cada
     *                        geração na saída padrão. Útil para tabuleiros pequenos,
     *                        proibitivo para execuções longas
     * @return o melhor indivíduo da última geração — ou o primeiro indivíduo ótimo
     *         encontrado, se o critério de parada por otimalidade disparar antes
     */
    public static Ind executar(int nPop, int nElite, IndFactory indFac, int nGer, boolean isMin, boolean imprimir_todos) {
        long tempo_inicial = System.currentTimeMillis();
        List<Ind> popIni = new ArrayList<>(nPop);
        List<Ind> newPop = new ArrayList<>(nPop);
        for (int i = 0; i < nPop; i++) {
            popIni.add(indFac.getInstance());
        }

        for (int g = 0; g < nGer; g++) {
            List<Ind> popIniAux = new ArrayList<>(nPop);
            popIniAux.addAll(popIni);

            List<Ind> popFilhos = new ArrayList<>(nPop);
            for (int i = 0; i < popIniAux.size() - 1; i += 2) {
                Ind pai1 = popIniAux.get(i);
                Ind pai2 = popIniAux.get(i + 1);

                List<Ind> newFilhos = pai1.recombinar(pai2);

                for (Ind filho : newFilhos) {
                    popFilhos.add(filho);
                }
            }

            List<Ind> popMutantes = new ArrayList<>(nPop);
            for (Ind ind : popIni) {
                popMutantes.add(ind.mutar());
            }

            List<Ind> join = new ArrayList<>(nPop * 3);
            join.addAll(popIni);
            join.addAll(popMutantes);
            join.addAll(popFilhos);
            boolean melhor_encontrado = false;

            newPop.clear();
            for (int i = 0; i < nElite; i++) {
                Ind melhor = melhorDe(join, isMin);
                if (melhor.getAvaliacao() == 0.0) {
                    melhor_encontrado = true;
                }
                newPop.add(melhor);
                join.remove(melhor);
            }

            RoletaViciada roleta = new RoletaViciada();
            for (Ind ind : roleta.roletar(join, nPop - nElite)) {
                newPop.add(ind);
            }

            popIni.clear();
            popIni.addAll(newPop);

            Ind melhor = melhorDe(popIni, isMin);
            if (imprimir_todos) {
                System.out.println("=============== GERACAO" + g + " ===============");
                for (Ind ind : popIni) {
                    System.out.println(ind);
                }
                System.out.println("\nMELHOR DA GERACAO G" + g + ": " + melhor + "\n==========================================\n\n");
            }

            if (melhor_encontrado) {
                long tempo_final = System.currentTimeMillis();
                System.out.println("INDIVIDUO OTIMO ENCONTRADO, EXECUCAO FINALIZADA COM " + ((double) (tempo_final - tempo_inicial) / 1000) + "s");
                return melhor;
            }
        }

        Ind melhor = melhorDe(popIni, isMin);
        long tempo_final = System.currentTimeMillis();
        System.out.println("EXECUCAO FINALIZADA COM " + ((double) (tempo_final - tempo_inicial) / 1000) + "s");
        return melhor;
    }

    /**
     * Varre uma lista e devolve o indivíduo de melhor avaliação, respeitando o sentido
     * da otimização.
     *
     * <p>Antes existia como três blocos de código idênticos dentro de {@link #executar}.
     * Extraí-lo também foi o que permitiu que o parâmetro {@code isMin} passasse a ter
     * efeito real: até então ele era recebido e ignorado, e todas as comparações eram
     * de minimização.</p>
     *
     * @param populacao lista não vazia de candidatos
     * @param isMin     {@code true} para tratar menor avaliação como melhor,
     *                  {@code false} para tratar maior avaliação como melhor
     * @return a referência ao melhor indivíduo da lista (não uma cópia)
     * @throws IndexOutOfBoundsException se {@code populacao} estiver vazia
     */
    private static Ind melhorDe(List<Ind> populacao, boolean isMin) {
        Ind melhor = populacao.get(0);
        for (int j = 1; j < populacao.size(); j++) {
            double atual = populacao.get(j).getAvaliacao();
            double referencia = melhor.getAvaliacao();
            if (isMin ? atual < referencia : atual > referencia) {
                melhor = populacao.get(j);
            }
        }
        return melhor;
    }
}
