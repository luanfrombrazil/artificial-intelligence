
/**
 * Ponto de entrada do projeto de algoritmos genéticos.
 *
 * <p>Configura os parâmetros da execução, escolhe o problema a ser resolvido através da
 * {@link IndFactory} correspondente e dispara o motor {@link Ag}.</p>
 *
 * <p><b>Reprodutibilidade.</b> A semente é fixada logo no início, antes de qualquer
 * sorteio, e impressa junto do resultado. Com a mesma semente e os mesmos parâmetros a
 * execução é bit a bit idêntica; para uma execução independente, troque a chamada por
 * {@link Rng#setSeedAleatoria()} — que também devolve a semente sorteada, de modo que a
 * execução continua registrável e repetível depois.</p>
 *
 * <p>Para trocar de problema, basta substituir a fábrica:</p>
 * <pre>{@code
 * IndFactory fabrica = new NRainhasIndFactory(8);      // tabuleiro 8x8
 * IndFactory fabrica = new DixonIndFactory(5);         // Dixon-Price em 5 dimensoes
 * IndFactory fabrica = new LangermannIndFactory(2);    // Langermann (somente 2D)
 * IndFactory fabrica = new PermIndFactory(4);          // Perm beta em 4 dimensoes
 * }</pre>
 */
public class AgMain {

    /**
     * Executa uma instância do algoritmo genético com os parâmetros definidos no corpo
     * do método.
     *
     * @param args não utilizado; os parâmetros são definidos no código
     */
    public static void main(String[] args) {
        // --- Reprodutibilidade -------------------------------------------------
        // Fixe a semente para repetir exatamente uma execução. Para uma execução
        // independente use Rng.setSeedAleatoria(), que devolve a semente sorteada.
        long semente = 42L;
        Rng.setSeed(semente);

        // --- Parâmetros da execução --------------------------------------------
        int num_pop = 20;                          // tamanho da população
        int qtd_elite = 8;                         // indivíduos que passam por elitismo
        int geracoes = 3000;                       // orçamento máximo de gerações
        int dimensao_problema = 8;                 // ordem do tabuleiro / dimensão
        boolean minimizar = true;                  // N-Rainhas minimiza colisões
        boolean imprimir_todos_individuos = false; // true só para tabuleiros pequenos

        IndFactory fabrica_individuos = new NRainhasIndFactory(dimensao_problema);

        System.out.println("SEMENTE: " + Rng.getSeed());
        Ind resultado = Ag.executar(num_pop, qtd_elite, fabrica_individuos,
                geracoes, minimizar, imprimir_todos_individuos);

        System.out.println("\n\nRESULTADO FINAL: " + resultado);
        if (resultado instanceof NRainhasInd) {
            System.out.println("SOLUCAO VALIDA: " + ((NRainhasInd) resultado).isValido());
        }
        System.out.println();
    }
}
