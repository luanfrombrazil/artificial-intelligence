package neural_networks;
import java.text.DecimalFormat;
import java.util.Random;

public class Main {
    public static double[][][] base_and = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {0.0}},
            {{1.0, 0.0}, {0.0}},
            {{1.0, 1.0}, {1.0}}
    };

    public static double[][][] base_or = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {1.0}},
            {{1.0, 0.0}, {1.0}},
            {{1.0, 1.0}, {1.0}}
    };

    public static double[][][] base_xor = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {1.0}},
            {{1.0, 0.0}, {1.0}},
            {{1.0, 1.0}, {0.0}}
    };

    public static double[][][] base_robo = {
            {{0.0, 0.0, 0.0}, {1.0, 1.0}},
            {{0.0, 0.0, 1.0}, {0.0, 1.0}},
            {{0.0, 1.0, 0.0}, {1.0, 0.0}},
            {{0.0, 1.0, 1.0}, {0.0, 1.0}},
            {{1.0, 0.0, 0.0}, {1.0, 0.0}},
            {{1.0, 0.0, 1.0}, {1.0, 0.0}},
            {{1.0, 1.0, 0.0}, {1.0, 0.0}},
            {{1.0, 1.0, 1.0}, {1.0, 0.0}}
    };

    public static void main(String[] args) {
        double[][][] base;
        /*
        PARA FACILITAR OS TESTES, FIZ DESSA MANEIRA, BASTA REMOVER O COMENTÁRIO DA LINHA DA BASE QUE SE DESEJA TESTAR
        */
        base = base_and;
        //base = base_or;
        //base = base_xor;
        //base = base_robo;

        /*
        PARA DEBUGAR O CÓDIGO COM FACILIDADE, ACRESCENTEI OS PARAMETROS FIXED AND VERBOSE, ESSES PARAMETROS MOSTRAM
        NO CONSOLE OS VALORES INICIAIS DE W0,W1,...,WN PARA QUE POSSA SER COMPARADO COM OS VALORES DA TABELA QUE FIZ
        NO GOOGLE SPREASHEETS.
         */
        Perceptron rna = new Perceptron(base[0].length, base[1].length,true,true);
        for (int epoca = 0; epoca <= 10000; epoca++) {
            double erro_epoca = 0;
            for (int amostra = 0; amostra < base.length; amostra++) {
                double[] x = base[amostra][0];
                double[] y = base[amostra][1];
                double out[] = rna.treinar(x, y);
                double erro_amostra = 0;
                for (int j = 0; j < y.length; j++) {
                    erro_amostra += Math.abs(y[j] - out[j]);
                }
                erro_epoca += erro_amostra;
            }
            System.out.println("Epoca "+epoca+" - error: "+erro_epoca);
        }
        for (int i = 0; i< base.length; i++){
            System.out.println(base[i][0][0]+" AND "+ base[i][0][1]+": "+rna.predict(base[i][0])[0]);
        }

    }
}

class Perceptron {
    private double[][] w;
    private int qtdIn;
    private int qtdOut;
    private double NI;
    private Random r;


    public Perceptron(int qtdIn, int qtdOut, boolean fixed, boolean verbose) {
        if (fixed) {
            /*COM O PARAMETRO FIXED, OS NUMEROS ALEATÓRIOS SÃO BASEADOS NA SEED 1*/
            this.qtdIn = qtdIn;
            this.qtdOut = qtdOut;
            this.NI = 0.3;
            this.w = new double[qtdIn + 1][qtdOut];
            this.r = new Random(1);
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    w[i][j] = r.nextDouble(0.6) - 0.3;
                }
            }
        } else {


            this.qtdIn = qtdIn;
            this.qtdOut = qtdOut;
            this.NI = 0.3;
            this.w = new double[qtdIn + 1][qtdOut];
            this.r = new Random();
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    w[i][j] = r.nextDouble(0.6) - 0.3;
                }
            }

        }


        if (verbose) {
            System.out.println("_____________________________________");
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    System.out.print(w[i][j]);
                }
                System.out.println();
            }
            System.out.println("_____________________________________");
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

    public double[]treinar(double[] x_in, double[] y_in) {
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
            out[j] = 1 / (1 + Math.exp(-u));
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
            out[j] = 1 / (1 + Math.exp(-u));

            if (out[j] > 0.5) {
                out[j] = 1;
            } else {
                out[j] = 0;
            }

        }
        return out;
    }
}

