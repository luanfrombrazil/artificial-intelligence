
import java.util.ArrayList;
import java.util.List;

public class Ag {

    public static Ind executar(int nPop, int nElite, IndFactory indFac, int nGer, boolean isMin) {

        List<Ind> popIni = new ArrayList(nPop);
        List<Ind> newPop = new ArrayList(nPop);
        for (int i = 0; i < nPop; i++) {
            popIni.add((NRainhasInd) indFac.getInstance());
        }

        for (int g = 0; g < nGer; g++) {
            List<Ind> popIniAux = new ArrayList<>(nPop);
            popIniAux.addAll(popIni);

            List<Ind> popFilhos = new ArrayList<>(nPop);
            for (int i = 0; i < popIniAux.size() - 1; i += 2) {
                NRainhasInd pai1 = (NRainhasInd) popIniAux.get(i);
                NRainhasInd pai2 = (NRainhasInd) popIniAux.get(i + 1);

                List<Ind> newFilhos = pai1.recombinar(pai2);

                for (Ind filho : newFilhos) {
                    popFilhos.add((NRainhasInd) filho);
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

            for (int i = 0; i < nElite; i++) {
                NRainhasInd melhor = (NRainhasInd) join.get(0);
                for (int j = 1; j < join.size(); j++) {
                    if (join.get(j).getAvaliacao() < melhor.getAvaliacao()) {
                        melhor = (NRainhasInd) join.get(j);
                    }
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
        }

        NRainhasInd melhor = (NRainhasInd) popIni.get(0);
        for (int j = 1; j < popIni.size(); j++) {
            if (popIni.get(j).getAvaliacao() < melhor.getAvaliacao()) {
                melhor = (NRainhasInd) popIni.get(j);
            }
        }
        return melhor;
    }
}
