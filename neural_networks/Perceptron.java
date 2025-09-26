package neural_networks;

import java.util.Random;

public class Perceptron {
    private double[][] w;
    private double ni;
    private Base base;
    private int epochs;
    private final int in;
    private final int out;
    public Statistics statistics;

    public Perceptron(Base base, int epochs, double ni, double random_range) {
        this.base = base;
        this.epochs = epochs > 0 ? epochs : 1000;
        this.ni = ni;
        this.in = base.getIn();
        this.out = base.getOut();
        this.w = new double[in + 1][out];
        Random random = new Random();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                w[i][j] = random.nextDouble(random_range * 2) - random_range;
            }
        }
    }

    public void train(boolean statistics) {
        if (statistics) this.statistics = new Statistics(epochs,base);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int line = 0; line < base.length; line++) {
                double[] x = new double[in + 1];
                x[in] = 1;
                for (int i = 0; i < in; i++) {
                    x[i] = this.base.getInput_data()[line][i];
                }

                double[] guess = new double[out];
                for (int j = 0; j < out; j++) {
                    double u = 0;
                    for (int i = 0; i < in + 1; i++) {
                        u += x[i] * w[i][j];
                    }
                    guess[j] = sigmoid(u);
                }

                if (statistics) {
                    boolean wrong_sample = false;
                    for (int i = 0; i < out; i++) {
                        double normalized = guess[i] > 0.5 ? 1 : 0;
                        if (normalized != base.getOutput_data()[line][i]) {
                            wrong_sample = true;
                            break;
                        }
                    }
                    this.statistics.append_classification(epoch, wrong_sample);
                }

                if (statistics) this.statistics.append_error(epoch, calculateError(line, guess));

                for (int j = 0; j < out; j++) {
                    double delta = (base.getOutput_data()[line][j] - guess[j]) * guess[j] * (1 - guess[j]);
                    for (int i = 0; i < in + 1; i++) {
                        w[i][j] += ni * delta * x[i];
                    }
                }
            }
            test(epoch, statistics);
        }
    }

    public void test(int epoch, boolean statistics) {
        if (!statistics) return;
        for (int line = 0; line < base.test_length; line++) {
            double[] x = new double[in + 1];
            x[in] = 1;
            for (int i = 0; i < in; i++) {
                x[i] = this.base.getTest_input_data()[line][i];
            }

            double[] guess = new double[out];
            for (int j = 0; j < out; j++) {
                double u = 0;
                for (int i = 0; i < in + 1; i++) {
                    u += x[i] * w[i][j];
                }
                guess[j] = sigmoid(u);
            }

            boolean wrong_sample = false;
            for (int i = 0; i < out; i++) {
                double normalized = guess[i] > 0.5 ? 1 : 0;
                if (normalized != base.getTest_output_data()[line][i]) {
                    wrong_sample = true;
                    break;
                }
            }
            this.statistics.append_test_classification_error(epoch, wrong_sample);
            this.statistics.append_test_error(epoch, calculateTestError(line, guess));
        }
    }

    public void status(boolean head, boolean body) {
        System.out.println(statistics.getData(head, body));
    }

    public double sigmoid(double u) {
        return 1 / (1 + Math.exp(-u));
    }

    public double calculateError(int line, double[] guess) {
        double total_error = 0.0;
        double[] expected_output = this.base.getOutput_data()[line];
        for (int j = 0; j < this.out; j++) {
            total_error += Math.abs(expected_output[j] - guess[j]);
        }
        return total_error;
    }

    public double calculateTestError(int line, double[] guess) {
        double total_error = 0.0;
        double[] expected_output = this.base.getTest_output_data()[line];
        for (int j = 0; j < this.out; j++) {
            total_error += Math.abs(expected_output[j] - guess[j]);
        }
        return total_error;
    }
}