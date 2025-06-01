
import java.util.Scanner;

public class AgMain {

    //TESTANDO O ALGORÍTMO GENÉTICO PARA N RAINHAS
    public static void main(String[] args) {
        Scanner sci = new Scanner(System.in);
        int num_pop = sci.nextInt();
        int qtd_elite = sci.nextInt();
        int geracoes = sci.nextInt();
        int dimensao_problema = sci.nextInt();
        boolean imprimir_todos_individuos = false;

        IndFactory fabrica_individuos = new DixonIndFactory(dimensao_problema);
        System.out.println("\n\n\nRESULTADO FINAL: " + Ag.executar(num_pop, qtd_elite, fabrica_individuos, geracoes, true, imprimir_todos_individuos) + "\n\n");

        fabrica_individuos = new PermIndFactory(dimensao_problema);
        System.out.println("\n\n\nRESULTADO FINAL: " + Ag.executar(num_pop, qtd_elite, fabrica_individuos, geracoes, true, imprimir_todos_individuos) + "\n\n");

        fabrica_individuos = new LangermannIndFactory(dimensao_problema);
        System.out.println("\n\n\nRESULTADO FINAL: " + Ag.executar(num_pop, qtd_elite, fabrica_individuos, geracoes, true, imprimir_todos_individuos) + "\n\n");
    }
}
