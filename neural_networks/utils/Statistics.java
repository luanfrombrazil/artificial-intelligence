package neural_networks.utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class Statistics {
    private final String baseName;
    private final List<Double> trainMseHistory = new ArrayList<>();
    private final List<Double> testMseHistory = new ArrayList<>();
    private final List<Integer> trainClassificationErrorHistory = new ArrayList<>();
    private final List<Integer> testClassificationErrorHistory = new ArrayList<>();

    public Statistics(String baseName) {
        this.baseName = baseName;
    }

    public void recordEpoch(double trainMse, int trainClassificationErrors, double testMse, int testClassificationErrors) {
        trainMseHistory.add(trainMse);
        trainClassificationErrorHistory.add(trainClassificationErrors);
        testMseHistory.add(testMse);
        testClassificationErrorHistory.add(testClassificationErrors);

        int epoch = trainMseHistory.size();


        if (epoch % 50 == 0)
            System.out.printf("Época: %-5d | Erros Treino: %-4d | Erros Teste: %-4d | MSE Treino: %.6f | MSE Teste: %.6f \n",
                    epoch, trainClassificationErrors, testClassificationErrors, trainMse, testMse);
        if (epoch == 500)
            System.out.println();
    }

    public void writeDetailedLog(int trainSize, int testSize) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
        String timestamp = LocalDateTime.now().format(formatter);
        String filename = String.format("data/[%s] %s.log", timestamp, baseName);

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
            int firstPerfectTrain = findFirstEpochWithZeroErrors(trainClassificationErrorHistory);
            int firstPerfectTest = findFirstEpochWithZeroErrors(testClassificationErrorHistory);
            int bestTestEpoch = findBestEpoch(testClassificationErrorHistory);
            int lastTrainErrors = trainClassificationErrorHistory.isEmpty() ? -1 : trainClassificationErrorHistory.get(trainClassificationErrorHistory.size() - 1);
            int lastTestErrors = testClassificationErrorHistory.isEmpty() ? -1 : testClassificationErrorHistory.get(testClassificationErrorHistory.size() - 1);

            bw.write("BASE: " + baseName + "\n");
            bw.write("PRIMEIRA EPOCA TREINO SEM ERRO: " + (firstPerfectTrain == -1 ? "NÃO CONVERGIU" : firstPerfectTrain) + "\n");
            bw.write("PRIMEIRA EPOCA TESTE SEM ERRO: " + (firstPerfectTest == -1 ? "NÃO CONVERGIU" : firstPerfectTest) + "\n");
            bw.write("MELHOR EPOCA DE TESTE: " + (bestTestEpoch == -1 ? "N/A" : bestTestEpoch) + "\n");
            bw.write("CLASSIFICOU TREINO CORRETAMENTE AO FINAL: " + (lastTrainErrors == 0 ? "SIM" : "NÃO") + "\n");
            bw.write("CLASSIFICOU TESTE CORRETAMENTE AO FINAL: " + (lastTestErrors == 0 ? "SIM" : "NÃO") + "\n");
            bw.write("\n\n");

            for (int i = 0; i < trainMseHistory.size(); i++) {
                if (i % 100 == 0) {
                    bw.write("============================================= RELATÓRIO DE TREINAMENTO =============================================\n");
                    bw.write(String.format("%-7s | %-12s | %-12s | %-10s | %-11s | %-10s | %-10s |\n", "EPOCA", "ERROS TREINO", "MSE TREINO", "ACC TREINO", "ERROS TESTE", "ACC TESTE", "MSE TESTE"));
                    bw.write("----------------------------------------------------------------------------------------------------------------------\n");
                }

                int trainErrors = trainClassificationErrorHistory.get(i);
                double trainMse = trainMseHistory.get(i);
                double trainAccuracy = ((double) trainSize - trainErrors) / trainSize;

                int testErrors = testClassificationErrorHistory.get(i);
                double testAccuracy = ((double) testSize - testErrors) / testSize;
                double testMse = testMseHistory.get(i);

                bw.write(String.format("%-7d | %-12d | %-12.8f | %-10.1f%% | %-11d | %-10.1f%% | %-10.8f |\n",
                        (i + 1),
                        trainErrors,
                        trainMse,
                        trainAccuracy * 100,
                        testErrors,
                        testAccuracy * 100,
                        testMse
                ));
            }
            bw.write("======================================================================================================================\n");

        } catch (IOException e) {
            System.err.println("Erro ao escrever log detalhado: " + filename);
            e.printStackTrace();
        }
    }

    public void exportToCSV() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
        String timestamp = LocalDateTime.now().format(formatter);
        String filename = String.format("data/[%s] %s.csv", timestamp, baseName);

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
            bw.write("epoch,train_mse,train_classification_errors,test_mse,test_classification_errors\n");

            for (int i = 0; i < trainMseHistory.size(); i++) {
                bw.write(String.format(Locale.US, "%d,%.8f,%d,%.8f,%d\n",
                        i + 1,
                        trainMseHistory.get(i),
                        trainClassificationErrorHistory.get(i),
                        testMseHistory.get(i),
                        testClassificationErrorHistory.get(i)
                ));
            }
        } catch (IOException e) {
            System.err.println("Erro ao exportar estatísticas para CSV: " + filename);
            e.printStackTrace();
        }
    }

    public void printFinalSummary() {
        int bestTestEpoch = -1;
        int minTestErrors = Integer.MAX_VALUE;

        for (int i = 0; i < testClassificationErrorHistory.size(); i++) {
            if (testClassificationErrorHistory.get(i) < minTestErrors) {
                minTestErrors = testClassificationErrorHistory.get(i);
                bestTestEpoch = i + 1;
            }
        }

        System.out.println("====== RESUMO DO TREINAMENTO ======");
        System.out.println("Base: " + baseName);
        System.out.println("Total de Épocas: " + trainMseHistory.size());
        if (bestTestEpoch != -1) {
            System.out.printf("Melhor época no conjunto de teste: %d (com %d erros)\n", bestTestEpoch, minTestErrors);
        }
        System.out.println("===================================");
    }

    private int findFirstEpochWithZeroErrors(List<Integer> errorHistory) {
        for (int i = 0; i < errorHistory.size(); i++) {
            if (errorHistory.get(i) == 0) return i + 1;
        }
        return -1;
    }

    private int findBestEpoch(List<Integer> testErrorHistory) {
        if (testErrorHistory.isEmpty()) return -1;

        int bestEpoch = 1;
        int minErrors = testErrorHistory.get(0);
        for (int i = 1; i < testErrorHistory.size(); i++) {
            if (testErrorHistory.get(i) < minErrors) {
                minErrors = testErrorHistory.get(i);
                bestEpoch = i + 1;
            }
        }
        return bestEpoch;
    }
}