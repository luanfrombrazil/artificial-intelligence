
import java.util.ArrayList;
import java.util.List;

public class Ag {

    public static Ind executar(int nPop, int nElite, IndFactory indFac, int nGer, boolean isMin, boolean imprimir_todos) {
        long tempo_inicial = System.currentTimeMillis();
        List<Ind> popIni = new ArrayList(nPop);
        List<Ind> newPop = new ArrayList(nPop);
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
                Ind melhor = join.get(0);
                for (int j = 1; j < join.size(); j++) {
                    if (join.get(j).getAvaliacao() < melhor.getAvaliacao()) {
                        melhor = join.get(j);
                    }
                }
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

            Ind melhor = popIni.get(0);
            for (int j = 1; j < popIni.size(); j++) {
                if (popIni.get(j).getAvaliacao() < melhor.getAvaliacao()) {
                    melhor = popIni.get(j);
                }
            }
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

        Ind melhor = popIni.get(0);
        for (int j = 1; j < popIni.size(); j++) {
            if (popIni.get(j).getAvaliacao() < melhor.getAvaliacao()) {
                melhor = popIni.get(j);
            }
        }
        long tempo_final = System.currentTimeMillis();
        System.out.println("EXECUCAO FINALIZADA COM " + ((double) (tempo_final - tempo_inicial) / 1000) + "s");
        return melhor;
    }
}
