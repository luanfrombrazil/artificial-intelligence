package neural_networks.core;

import neural_networks.activations.ActivationFunction;
import neural_networks.utils.Base;
import neural_networks.utils.Statistics;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();
    private final int epochs;
    private final double learningRate;
    private final Base base;
    public Statistics statistics;

    public NeuralNetwork(Base base, int[] architecture, ArrayList<ActivationFunction> functions, int epochs, double learningRate, double randomRange) {
        this.base = base;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.statistics = new Statistics(base.getName());

        for (int i = 0; i < architecture.length - 1; i++) {
            if (i == architecture.length - 1)
                layers.add(new Layer(architecture[i], architecture[i + 1], functions.get(i), randomRange, false));
            else
                layers.add(new Layer(architecture[i], architecture[i + 1], functions.get(i), randomRange, true));
        }
    }

    public double[] predict(double[] inputs) {
        double[] currentOutput = inputs;
        for (Layer layer : layers) {
            currentOutput = layer.forward(currentOutput);
        }
        return currentOutput;
    }

    public void train() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochTrainTotalError = 0.0;
            int epochTrainClassificationErrors = 0;

            for (int i = 0; i < base.length; i++) {
                double[] input = base.getInput_data()[i];
                double[] target = base.getOutput_data()[i];

                double[] prediction = predict(input);

                double[] delta = new double[target.length];
                for (int j = 0; j < target.length; j++) {
                    delta[j] = target[j] - prediction[j];
                }

                for (int j = layers.size() - 1; j >= 0; j--) {
                    delta = layers.get(j).backward(delta);
                }

                for (Layer layer : layers) {
                    layer.updateParameters(learningRate);
                }
                epochTrainTotalError += calculateSampleError(target, prediction);
                if (wasMisclassified(target, prediction, false)) { // 'false' indica que é amostra de treino
                    epochTrainClassificationErrors++;
                }
            }

            double[] testResults = test();
            double epochTestTotalError = testResults[0];
            int epochTestClassificationErrors = (int) testResults[1];

            double trainMse = epochTrainTotalError / base.length;
            double testMse = epochTestTotalError / base.test_length;

            statistics.recordEpoch(trainMse, epochTrainClassificationErrors, testMse, epochTestClassificationErrors);
        }
        System.out.println();
    }

    public double[] test() {
        double totalTestError = 0.0;
        int classificationErrors = 0;

        for (int i = 0; i < base.test_length; i++) {
            double[] input = base.getTest_input_data()[i];
            double[] target = base.getTest_output_data()[i];
            double[] prediction = predict(input);

            totalTestError += calculateSampleError(target, prediction);
            if (wasMisclassified(target, prediction, true)) {
                classificationErrors++;
            }
        }
        return new double[]{totalTestError, classificationErrors};
    }

    private double calculateSampleError(double[] target, double[] prediction) {
        double error = 0;
        for (int i = 0; i < target.length; i++) {
            error += Math.abs(target[i] - prediction[i]);
        }
        return error;
    }

    private boolean wasMisclassified(double[] target, double[] prediction, boolean isTest) {
        double maior = -1;
        for (int i = 0; i < prediction.length; i++) {
            if (prediction[i] > maior)
                maior = prediction[i];
        }
        boolean wrong_sample = false;
        double thresholdValue = maior;

        for (int i = 0; i < prediction.length; i++) {
            double normalized;
            if (target[i] % 1 == 0)
                normalized = prediction[i] >= thresholdValue ? Math.ceil(prediction[i]) : Math.floor(prediction[i]);
            else
                normalized = prediction[i] >= thresholdValue ? Math.ceil(prediction[i]) - 0.005 : Math.floor(prediction[i]) + 0.005;

            if (normalized != target[i]) {
                wrong_sample = true;
                break;
            }
        }
        return wrong_sample;
    }

    public void printLog() {
        statistics.writeDetailedLog(base.length, base.test_length);
    }

    public void generateCSV() {
        statistics.exportToCSV();
    }
}