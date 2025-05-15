
import java.util.Scanner;

public class AgMain {

    //TESTANDO O ALGORÍTMO GENÉTICO PARA N RAINHAS
    public static void main(String[] args) {
        /*
        
        BOM DIA, DOUGLAS.
        SEGUE A LINHA DE COMANDO PARA COMPILAR E EXECUTAR RECEBENDO COMO INPUT OS PARAMETROS DO ARQUIVO parameters.in :

        cmd>    javac -d out AgMain.java ; rm log.out ; cd out ; cat ..\parameters.in | java AgMain > ..\log.out ; cd ..

        O ARQUIVO CONTEM OS PARAMETROS PARA UMA POPULAÇÃO DE 60 INDIVIDUOS, DOS QUAIS 20 SAO OBTIDOS VIA ELITISMO E 5 MIL GERAÇÕES. 
        TENTARÃO RESOLVER UM PROBLEMA PARA 12 RAINHAS.
        
        TODO OUTPUT DO TIPO PRINT ESTÁ NO ARQUIVO log.out, O ARQUIVO MOSTRA TODA A POPULAÇÃO RESULTANTE DE CADA GERAÇÃO E SEU RESPECTIVO INDIVÍDUO COM MELHOR AVALIAÇÃO.



        OBS.:
        CASO RODE POR UMA IDE, OS VALORES DEVEM SER INSERIDOS NA ORDEM:
        NUMERO DE INDIVÍDUOS DA POPULAÇÃO
        NUMERO DE INDIVIDUOS OBTIDOS VIA ELITISMO
        NUMERO DE GERAÇÕES
        NUMERO DE RAINHAS/TAMANHO DO PROBLEMA.
        
        E NÃO SERÃO COLOCADOS NO ARQUIVO log.out
        */

        Scanner sci = new Scanner(System.in);
        int n = sci.nextInt();
        int e = sci.nextInt();
        int g = sci.nextInt();
        IndFactory indFac = new NRainhasIndFactory(sci.nextInt());
        System.out.println("\n\n\nRESULTADO FINAL: " + Ag.executar(n, e, indFac, g, true) + "\n\n");
    }
}
