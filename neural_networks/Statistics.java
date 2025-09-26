package neural_networks;

public class Statistics {
    private Base base;
    private double[] error;
    private int[] classification_error;
    private double[] test_error;
    private int[] test_classification_error;

    Statistics(int epochs, Base base) {
        this.base = base;
        this.error = new double[epochs];
        this.classification_error = new int[epochs];
        this.test_error = new double[epochs];
        this.test_classification_error = new int[epochs];
    }

    void append_error(int epoch, double error) {
        this.error[epoch] += error;
    }

    void append_classification(int epoch, boolean incorrect) {
        if (incorrect) this.classification_error[epoch]++;
    }

    void append_test_error(int epoch, double error) {
        this.test_error[epoch] += error;
    }

    void append_test_classification_error(int epoch, boolean incorrect) {
        if (incorrect) this.test_classification_error[epoch]++;
    }

    public int findFirstPerfectTrainEpoch() {
        for (int i = 0; i < classification_error.length; i++) {
            if (classification_error[i] == 0) return i + 1;
        }
        return -1;
    }

    public int findFirstPerfectTestEpoch() {
        for (int i = 0; i < test_classification_error.length; i++) {
            if (test_classification_error[i] == 0) return i + 1;
        }
        return -1;
    }

    public String getData(boolean head, boolean body) {
        StringBuilder sb = new StringBuilder();
        if (body) {
            for (int i = 0; i < error.length; i++) {
                if (i % 100 == 0) {
                    sb.append("================================================= RELATÓRIO DE TREINAMENTO (PERCEPTRON SIMPLES) =================================================\n");
                    sb.append(String.format("%-7s | %-15s | %-20s | %-15s | %-15s | %-15s | %-15s |\n", "EPOCA", "ERROS", "MSE", "ACURÁCIA TREINO ", "ERROS TESTE", "TESTE MSE ", "ACURÁCIA TESTE  "));
                    sb.append("---------------------------------------------------------------------------------------------------------------------------\n");
                }
                double mse = error[i] / base.length;
                double accuracy = ((double) base.length - classification_error[i]) / base.length;
                double test_mse = test_error[i] / base.test_length;
                double test_accuracy = ((double) base.test_length - test_classification_error[i]) / base.test_length;
                sb.append(String.format("%-7d | %-15d | %-20.8f | %-15.1f%% | %-15d | %-15.8f | %-15.1f%% |\n",
                        (i + 1),
                        classification_error[i],
                        mse,
                        accuracy * 100,
                        test_classification_error[i],
                        test_mse,
                        test_accuracy * 100
                ));
            }
            sb.append("============================================================================================================================\n");
        }
        if (head) {
            sb.append("BASE: " + base.name + "\n");
            sb.append("PRIMEIRA EPOCA SEM ERRO DE CLASSIFICAÇÃO: " + findFirstPerfectTrainEpoch() + "\n");
            sb.append("PRIMEIRA EPOCA TESTE SEM ERRO DE CLASSIFICAÇÃO: " + findFirstPerfectTestEpoch() + "\n");
            sb.append("CLASSIFICOU CORRETAMENTE TODAS AMOSTRAS: " + ((classification_error.length > 0 && classification_error[classification_error.length - 1] == 0) ? "SIM\n" : "NÃO\n"));
            sb.append("CLASSIFICOU CORRETAMENTE TODAS AMOSTRAS DE TESTE: " + ((test_classification_error.length > 0 && test_classification_error[test_classification_error.length - 1] == 0) ? "SIM" : "NÃO"));
        }
        return sb.toString();
    }
}