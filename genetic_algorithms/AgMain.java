
import java.util.Scanner;

public class AgMain {

    //TESTANDO O ALGORÍTMO GENÉTICO PARA N RAINHAS
    public static void main(String[] args) {
        Scanner sci = new Scanner(System.in);
        int num_pop = 20;
        int qtd_elite = 8;
        int geracoes = 3000;
        int dimensao_problema = 4;
        boolean imprimir_todos_individuos = true;

        IndFactory fabrica_individuos = new NRainhasIndFactory(dimensao_problema);
        System.out.println("\n\n\nRESULTADO FINAL: " + Ag.executar(num_pop, qtd_elite, fabrica_individuos, geracoes, true, imprimir_todos_individuos) + "\n\n");
    }
}
