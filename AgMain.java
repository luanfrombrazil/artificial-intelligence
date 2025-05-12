
public class AgMain {


    //TESTANDO O ALGORÍTMO GENÉTICO PARA N RAINHAS
    public static void main(String[] args) {
        int nPop = 40;
        int nElite = 8;
        int nGer = 500;
        IndFactory indFac = new NRainhasIndFactory(10);

        for (int i = 0; i < 10; i++) {
            Ind melhor = Ag.executar(nPop, nElite, indFac, nGer, true);
            System.out.println(melhor);
        }
        System.out.println();
        System.out.println();
    }
}
