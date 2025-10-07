package neural_networks.utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Statistics {
    private Base base;
    private double[] error;
    private int[] classification_error;
    private double[] test_error;
    private int[] test_classification_error;

    public Statistics(int epochs, Base base) {
        this.base = base;
        this.error = new double[epochs];
        this.classification_error = new int[epochs];
        this.test_error = new double[epochs];
        this.test_classification_error = new int[epochs];
    }

    public void append_error(int epoch, double error) {
        this.error[epoch] += error;
    }

    public void append_classification(int epoch, boolean incorrect) {
        if (incorrect) this.classification_error[epoch]++;
    }

    public void append_test_error(int epoch, double error) {
        this.test_error[epoch] += error;
    }

    public void append_test_classification_error(int epoch, boolean incorrect) {
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

    public int findBestTestEpoch() {
        int menor = test_classification_error[0];
        for (int i = 1; i < test_classification_error.length; i++) {
            if (test_classification_error[i] > menor)
                return i+1;
        }
//        for (int i = test_classification_error.length - 1; i >= 0; i--) {
//            if (test_classification_error[i] == menor) return i + 1;
//        }
        return -1;
    }

    public void printData(boolean head, boolean body) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
        String timestamp = LocalDateTime.now().format(formatter);
        String filename = "[" + timestamp + "] " + base.name + ".log";

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename, true))) {
            if (head) {
                bw.write("BASE: " + base.name + "\n");
                bw.write("PRIMEIRA EPOCA SEM ERRO DE CLASSIFICAÇÃO: " + (findFirstPerfectTrainEpoch() == -1 ? "NÃO CONVERGIU TOTALMENTE" : findFirstPerfectTrainEpoch()) + "\n");
                bw.write("PRIMEIRA EPOCA TESTE SEM ERRO DE CLASSIFICAÇÃO: " + (findFirstPerfectTestEpoch() == -1 ? "NÃO CONVERGIU TOTALMENTE" : findFirstPerfectTestEpoch()) + "\n");
                bw.write("MELHOR EPOCA DE TREINO: " + findBestTestEpoch() + "\n");
                bw.write("CLASSIFICOU CORRETAMENTE TODAS AMOSTRAS: " + ((classification_error.length > 0 && classification_error[classification_error.length - 1] == 0) ? "SIM\n" : "NÃO\n"));
                bw.write("CLASSIFICOU CORRETAMENTE TODAS AMOSTRAS DE TESTE: " + ((test_classification_error.length > 0 && test_classification_error[test_classification_error.length - 1] == 0) ? "SIM\n" : "NÃO\n"));
            }
            bw.write("\n\n");
            if (body) {
                for (int i = 0; i < error.length; i++) {
                    if (i % 100 == 0) {
                        bw.write("==================== RELATÓRIO DE TREINAMENTO (PERCEPTRON SIMPLES) =====================\n");
                        bw.write(String.format("%-7s | %-7s | %-11s | %-10s | %-11s | %-10s | %-10s |\n", "EPOCA", "ERROS", "MSE", "ACC TREINO ", "ERROS TESTE", "TESTE MSE ", "ACC TESTE  "));
                        bw.write("----------------------------------------------------------------------------------------\n");
                    }

                    double mse = error[i] / base.length;
                    double accuracy = ((double) base.length - classification_error[i]) / base.length;
                    double test_mse = test_error[i] / base.test_length;
                    double test_accuracy = ((double) base.test_length - test_classification_error[i]) / base.test_length;
                    bw.write(String.format("%-7d | %-7d | %-11.8f | %-10.1f%% | %-11d | %-10.8f | %-10.1f%% |\n",
                            (i + 1),
                            classification_error[i],
                            mse,
                            accuracy * 100,
                            test_classification_error[i],
                            test_mse,
                            test_accuracy * 100
                    ));
                }
                bw.write("========================================================================================\n");

            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}