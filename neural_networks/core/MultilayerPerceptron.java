package neural_networks.core;

import neural_networks.utils.Base;
import neural_networks.utils.Statistics;

import java.util.Random;

public class MultilayerPerceptron {
    private double[][] wh;
    private double[][] w_theta;
    private double ni;
    private Random random;
    private Base base;

    private int epochs;

    private final int in;
    private final int out;
    private final int hidden;

    public Statistics statistics;


    public void printLog() {
        statistics.writeDetailedLog(base.length, base.test_length);
    }

    public void generateCSV() {
        statistics.exportToCSV();
    }

    public MultilayerPerceptron(Base base, int epochs, double ni, double random_range) {
        this.base = base;
        this.ni = ni;

        this.hidden = base.getHidden();
        this.in = base.getIn();
        this.out = base.getOut();

        this.statistics = new Statistics(base.getName());
        this.epochs = epochs;
        this.random = new Random();

        this.wh = new double[in + 1][hidden];
        this.w_theta = new double[hidden + 1][out];

        for (int i = 0; i < wh.length; i++) {
            for (int j = 0; j < wh[0].length; j++) {
                wh[i][j] = random.nextDouble(random_range * 2) - random_range;
            }
        }
        for (int i = 0; i < w_theta.length; i++) {
            for (int j = 0; j < w_theta[0].length; j++) {
                w_theta[i][j] = random.nextDouble(random_range * 2) - random_range;
            }
        }
    }

    public void train() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochTrainTotalError = 0.0;
            int epochTrainClassificationErrors = 0;
            double[] guess = new double[out];
            for (int line = 0; line < base.length; line++) {
                double[] x = new double[in + 1];
                x[in] = 1;
                for (int i = 0; i < in; i++) {
                    x[i] = this.base.getInput_data()[line][i];
                }

                double[] hidden_layer = new double[hidden + 1];

                for (int h = 0; h < wh[0].length; h++) {
                    double u = 0;
                    for (int i = 0; i < wh.length; i++) {
                        u += x[i] * wh[i][h];
                    }
                    hidden_layer[h] = sigmoid(u);
                }
                hidden_layer[hidden_layer.length - 1] = 1;


                for (int j = 0; j < w_theta[0].length; j++) {
                    double u = 0;
                    for (int h = 0; h < w_theta.length; h++) {
                        u += hidden_layer[h] * w_theta[h][j];
                    }
                    guess[j] = sigmoid(u);
                }

                if (wrong_sample(guess, line)) {
                    epochTrainClassificationErrors++;
                }
                epochTrainTotalError += calculateError(line, guess);

                double[] delta_theta = new double[out];
                for (int j = 0; j < out; j++) {
                    delta_theta[j] = (base.getOutput_data()[line][j] - guess[j]) * guess[j] * (1 - guess[j]) * Math.abs((1 - guess[j]));
                }

                double[] delta_h = new double[hidden];
                for (int h = 0; h < hidden; h++) {
                    double soma = 0;
                    for (int j = 0; j < out; j++) {

                        soma += delta_theta[j] * w_theta[h][j];
                    }

                    delta_h[h] = soma * hidden_layer[h] * (1 - hidden_layer[h]);
                }

                for (int j = 0; j < out; j++) {
                    for (int h = 0; h < hidden + 1; h++) {
                        w_theta[h][j] += ni * delta_theta[j] * hidden_layer[h];
                    }
                }

                for (int h = 0; h < hidden; h++) {
                    for (int i = 0; i < in + 1; i++) {
                        wh[i][h] += ni * delta_h[h] * x[i];
                    }
                }
            }
            double[] testResults = test();
            double epochTestTotalError = testResults[0];
            int epochTestClassificationErrors = (int) testResults[1];

            double trainMse = epochTrainTotalError / base.length;
            double testMse = epochTestTotalError / base.test_length;

            statistics.recordEpoch(trainMse, epochTrainClassificationErrors, testMse, epochTestClassificationErrors);

        }
    }

    private boolean wrong_sample(double[] guess, int line) {
        double maior = -1;
        for (int i = 0; i < guess.length; i++) {
            if (guess[i] > maior)
                maior = guess[i];
        }
        boolean wrong_sample = false;
        double thresholdValue = maior;

        for (int i = 0; i < out; i++) {
            double normalized;
            if (base.getOutput_data()[line][i] % 1 == 0)
                normalized = guess[i] >= thresholdValue ? Math.ceil(guess[i]) : Math.floor(guess[i]);
            else
                normalized = guess[i] >= thresholdValue ? Math.ceil(guess[i]) - 0.005 : Math.floor(guess[i]) + 0.005;

            if (normalized != base.getOutput_data()[line][i]) {
                wrong_sample = true;
                break;
            }
        }
        return wrong_sample;
    }

    public double[] test() {
        double totalTestError = 0.0;
        int classificationErrors = 0;

        for (int line = 0; line < base.test_length; line++) {
            double[] x = new double[in + 1];
            x[in] = 1;
            for (int i = 0; i < in; i++) {
                x[i] = this.base.getTest_input_data()[line][i];
            }
            double[] hidden_layer = new double[hidden + 1];
            for (int h = 0; h < wh[0].length; h++) {
                double u = 0;
                for (int i = 0; i < wh.length; i++) {
                    u += x[i] * wh[i][h];
                }
                hidden_layer[h] = sigmoid(u);
            }
            hidden_layer[hidden_layer.length - 1] = 1;

            double[] guess = new double[out];
            for (int j = 0; j < w_theta[0].length; j++) {
                double u = 0;
                for (int h = 0; h < w_theta.length; h++) {
                    u += hidden_layer[h] * w_theta[h][j];
                }
                guess[j] = sigmoid(u);
            }

            if (wrong_test_sample(guess, line)) {
                classificationErrors++;
            }
            totalTestError += calculateTestError(line, guess);
        }

        return new double[]{totalTestError, classificationErrors};
    }

    private boolean wrong_test_sample(double[] guess, int line) {
        double maior = -1;
        for (int i = 0; i < guess.length; i++) {
            if (guess[i] > maior)
                maior = guess[i];
        }
        boolean wrong_sample = false;
        double thresholdValue = maior;

        for (int i = 0; i < out; i++) {
            double normalized;
            if (base.getTest_output_data()[line][i] % 1 == 0)
                normalized = guess[i] >= thresholdValue ? Math.ceil(guess[i]) : Math.floor(guess[i]);
            else
                normalized = guess[i] >= thresholdValue ? Math.ceil(guess[i]) - 0.005 : Math.floor(guess[i]) + 0.005;

            if (normalized != base.getTest_output_data()[line][i]) {
                wrong_sample = true;
                break;
            }
        }
        return wrong_sample;
    }


    public double sigmoid(double u) {
        return 1 / (1 + Math.exp(-u));
    }

    public double calculateError(int line, double[] guess) {
        double total_error = 0.0;
        double[] expected_output = this.base.getOutput_data()[line];
        for (int j = 0; j < this.out; j++) {
            double erro = Math.abs(expected_output[j] - guess[j]);
            total_error += erro;
        }
        return total_error;
    }

    public double calculateTestError(int line, double[] guess) {
        double total_error = 0.0;
        double[] expected_output = this.base.getTest_output_data()[line];
        for (int j = 0; j < this.out; j++) {
            double erro = Math.abs(expected_output[j] - guess[j]);
            total_error += erro;
        }
        return total_error;
    }
}
