package neural_networks;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Scanner;

public class Main {
    public static double[][][] AND = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {0.0}},
            {{1.0, 0.0}, {0.0}},
            {{1.0, 1.0}, {1.0}}
    };

    public static double[][][] OR = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {1.0}},
            {{1.0, 0.0}, {1.0}},
            {{1.0, 1.0}, {1.0}}
    };

    public static double[][][] XOR = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {1.0}},
            {{1.0, 0.0}, {1.0}},
            {{1.0, 1.0}, {0.0}}
    };

    public static double[][][] ROBO = {
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

        double[][][] TICTAC = new double[958][2][];
        try (Scanner sc = new Scanner(new File("tic-tac-toe.data"))) {
            for (int i = 0; i < 958; i++) {
                double[] tabuleiro = new double[9];
                for (int j = 0; j < 9; j++) {
                    tabuleiro[j] = sc.nextDouble();
                }
                TICTAC[i][0] = tabuleiro;

                double[] resultado = new double[2];
                resultado[0] = sc.nextDouble();
                resultado[1] = sc.nextDouble();
                TICTAC[i][1] = resultado;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[][][] INFLAMATORY = new double[120][2][];
        try (Scanner sc = new Scanner(new File("inflamatory.data"))) {
            for (int i = 0; i < 120; i++) {
                double[] INPUT = new double[6];
                for (int j = 0; j < 6; j++) {
                    INPUT[j] = sc.nextDouble();
                }
                INFLAMATORY[i][0] = INPUT;

                double[] RESULTADO = new double[4];
                RESULTADO[0] = 0.0;
                RESULTADO[1] = sc.nextDouble();
                RESULTADO[2] = 1.0;
                RESULTADO[3] = sc.nextDouble();

                INFLAMATORY[i][1] = RESULTADO;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        base = TICTAC;

        MultilayerPerceptron rna = new MultilayerPerceptron(base[0][0].length, base[0][1].length,0.01);
        //Perceptron rna = new Perceptron(base[0][0].length, base[0][1].length);

        for (int epoca = 0; epoca <= 10000; epoca++) {
            double erro_epoca = 0;
            int erro_classificacao_epoca = 0;
            for (int amostra = 0; amostra < base.length; amostra++) {
                double[] x = base[amostra][0];
                double[] y = base[amostra][1];
                double out[] = rna.treinar(x, y);
                double erro_amostra = 0;
                boolean classificou_errado = false;
                for (int j = 0; j < y.length; j++) {
                    erro_amostra += Math.abs(y[j] - out[j]);
                    out[j] = out[j] > 0.5 ? 1 : 0;
                    if (y[j] != out[j]) {
                        classificou_errado = true;
                    }
                }
                erro_epoca += erro_amostra;
                if (classificou_errado) {
                    erro_classificacao_epoca++;
                }
            }
            System.out.println("Epoca " + epoca + " - erro total: " + erro_epoca + " - erro de classificação: " + erro_classificacao_epoca);
        }
        DecimalFormat dc = new DecimalFormat("#.##");
    }
}


