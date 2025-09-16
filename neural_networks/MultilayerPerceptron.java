package neural_networks;

import java.util.Random;

public class MultilayerPerceptron {
    private double[][] wh;
    private double[][] w_theta;
    private int qtdIn;
    private int qtdOut;
    private int qtdHidden;
    private double ni;
    private Random random;

    public MultilayerPerceptron(int qtdIn, int qtdHidden, int qtdOut, double ni) {
        this.qtdIn = qtdIn;
        this.qtdOut = qtdOut;
        this.qtdHidden = qtdHidden;
        this.random = new Random();
        
        this.wh = new double[qtdIn + 1][qtdHidden];
        this.w_theta = new double[qtdHidden + 1][qtdOut];
        this.ni = ni;
        for (int i = 0; i < wh.length; i++) {
            for (int j = 0; j < wh[0].length; j++) {
                wh[i][j] = random.nextDouble(0.6) - 0.3;
            }
        }
        for (int i = 0; i < w_theta.length; i++) {
            for (int j = 0; j < w_theta[0].length; j++) {
                w_theta[i][j] = random.nextDouble(0.6) - 0.3;
            }
        }
    }

    public MultilayerPerceptron(int qtdIn, int qtdOut, double ni) {
        this(qtdIn,(qtdIn + qtdOut) / 2,qtdOut,ni);
    }

    public double[] treinar(double[] x_in, double[] y) {
        double[] x = new double[x_in.length + 1];
        x[x_in.length] = 1;
        for (int i = 0; i < x_in.length; i++) {
            x[i] = x_in[i];
        }

        double[] hidden = new double[qtdHidden + 1];

        for (int h = 0; h < wh[0].length; h++) {
            double u = 0;
            for (int i = 0; i < wh.length; i++) {
                u += x[i] * wh[i][h];
            }
            hidden[h] = sigmoid(u);
        }
        hidden[hidden.length - 1] = 1;

        double[] out = new double[qtdOut];
        for (int j = 0; j < w_theta[0].length; j++) {
            double u = 0;
            for (int h = 0; h < w_theta.length; h++) {
                u += hidden[h] * w_theta[h][j];
            }
            out[j] = sigmoid(u);
        }

        double[] delta_theta = new double[qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            delta_theta[j] = out[j] * (1 - out[j]) * (y[j] - out[j]);
        }
        double[] delta_h = new double[qtdHidden];
        for (int h = 0; h < qtdHidden; h++) {
            delta_h[h] = hidden[h] * (1 - hidden[h]);
            double soma = 0;
            for (int j = 0; j < qtdOut; j++) {
                soma += delta_theta[j] * w_theta[h][j];
            }
            delta_h[h] *= soma;
        }

        for (int j = 0; j < w_theta[0].length; j++) {
            for (int h = 0; h < w_theta.length; h++) {
                w_theta[h][j] += ni*delta_theta[j]*hidden[h];
            }
        }
        for (int h = 0; h < wh[0].length; h++) {
            for (int i = 0; i < wh.length; i++) {
                wh[i][h] += ni*delta_h[h]*x[i];
            }
        }
        return out;
    }

    public double ReLU(double u) {
        return Math.max(0, u);
    }

    public double sigmoid(double u) {
        return 1 / (1 + Math.exp(-u));
    }

    public double tanh(double u) {
        return Math.tanh(u);
    }

    public double leakyReLU(double u) {
        double alpha = 0.01;
        return u > 0 ? u : alpha * u;
    }

    public double ELU(double u) {
        double alpha = 1.0;
        return u > 0 ? u : alpha * (Math.exp(u) - 1);
    }
}
