package neural_networks;

import java.text.DecimalFormat;
import java.util.Random;

class Perceptron {
    private double[][] w;
    private int qtdIn;
    private int qtdOut;
    private double NI;
    private Random r;


    public Perceptron(int qtdIn, int qtdOut) {
        this.qtdIn = qtdIn;
        this.qtdOut = qtdOut;
        this.NI = .3;
        this.w = new double[qtdIn + 1][qtdOut];
        this.r = new Random();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                w[i][j] = r.nextDouble(0.6) - 0.3;
            }
        }
    }

    public Perceptron(int qtdIn, int qtdOut, double ni) {
        this.qtdIn = qtdIn;
        this.qtdOut = qtdOut;
        this.NI = ni;
        this.w = new double[qtdIn + 1][qtdOut];
        this.r = new Random();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                w[i][j] = r.nextDouble(0.6) - 0.3;
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        DecimalFormat df = new DecimalFormat("#.##");
        sb.append("P:{\n");
        for (double[] darray : this.w) {
            sb.append("{");
            for (double d : darray) {
                sb.append(df.format(d) + ", ");
            }
            sb.append("}\n");
        }
        sb.append("}");
        return sb.toString();
    }

    public double[] treinar(double[] x_in, double[] y_in) {
        double[] x = new double[x_in.length + 1];
        x[0] = 1;
        for (int i = 1; i < x.length; i++) {
            x[i] = x_in[i - 1];
        }
        double[] out = new double[this.qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            double u = 0;
            for (int i = 0; i < qtdIn + 1; i++) {
                u += x[i] * w[i][j];
            }
            /*
            UTILIZANDO A FUNÇÃO RELU PARA CALCULAR O ERRO E TREINAR A REDE NEURAL
            */

            out[j] = sigmoid(u);
            //out[j] = ReLU(u);
            //out[j] = tanh(u);
            //out[j] = leakyReLU(u);
            //out[j] = ELU(u);
            //out[j] = u;
        }
        double[][] delta_w = new double[w.length][w[0].length];
        for (int j = 0; j < qtdOut; j++) {
            for (int i = 0; i < qtdIn + 1; i++) {
                delta_w[i][j] = this.NI * (y_in[j] - out[j]) * x[i];
                w[i][j] += delta_w[i][j];
            }
        }
        return out;
    }


    /*
    ESCREVI ESSA FUNÇÃO APENAS PARA TESTAR O PERCEPTRON DEPOIS DO TREINAMENTO NAS 10K EPOCAS.
    DADO UMA ENTRADA X_IN ELE RETORNA UM VALOR.

    FIZ DE FORMA QUE O OUTPUT RETORNASSE APENAS 1 E 0, RESPEITANDO O TREINAMENTO DAS PORTAS LÓGICAS AND E OR
    */
    public double[] predict(double[] x_in) {
        double[] x = new double[x_in.length + 1];
        x[0] = 1;
        for (int i = 1; i < x.length; i++) {
            x[i] = x_in[i - 1];
        }
        double[] out = new double[this.qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            double u = 0;
            for (int i = 0; i < qtdIn + 1; i++) {
                u += x[i] * w[i][j];
            }
            out[j] = sigmoid(u);
            //out[j] = ReLU(u);
            //out[j] = tanh(u);
            //out[j] = leakyReLU(u);
            //out[j] = ELU(u);
            //out[j] = u;
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
