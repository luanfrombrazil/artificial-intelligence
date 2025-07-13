
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RoletaViciada {

    private static final Random random = new Random();

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
            double r = random.nextDouble();
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
