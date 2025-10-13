package neural_networks.utils;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

public class Base {
    private int in;
    private int out;
    private int hidden;

    public int length;
    private double[][] input_data;
    private double[][] output_data;

    public int test_length;
    private double[][] test_input_data;
    private double[][] test_output_data;

    private int numTrainSamples;
    private int numTestSamples;


    private String name;

    public String getName() {
        return name;
    }

    // No arquivo Base.java
    public Base(String name) {
        this.name = name.toUpperCase();
        String trainPath = "bases/datasets/" + name + "/" + name + ".train";

        // Bloco Try-Catch para o arquivo de TREINO
        try (Scanner sc = new Scanner(new File(trainPath))) {
            this.length = sc.nextInt();
            this.in = sc.nextInt();
            this.out = sc.nextInt();
            this.hidden = (in + out) / 2 + 2;

            this.input_data = new double[length][in];
            this.output_data = new double[length][out];

            for (int l = 0; l < length; l++) {
                for (int j = 0; j < in; j++) {
                    input_data[l][j] = sc.nextDouble();
                }
                for (int j = 0; j < out; j++) {
                    output_data[l][j] = sc.nextDouble();
                }
            }
        } catch (IOException e) {
            System.err.println("ERRO AO LER ARQUIVO DE TREINO: " + trainPath);
            e.printStackTrace();
        }

        // Bloco Try-Catch SEPARADO para o arquivo de TESTE
        String testPath = "bases/datasets/" + name + "/" + name + ".test";
        try (Scanner ts = new Scanner(new File(testPath))) {
            this.test_length = ts.nextInt();
            // Pula a leitura de in/out do arquivo de teste, pois já foram definidos
            ts.nextInt();
            ts.nextInt();

            this.test_input_data = new double[test_length][in];
            this.test_output_data = new double[test_length][out];

            for (int l = 0; l < test_length; l++) {
                for (int j = 0; j < in; j++) {
                    test_input_data[l][j] = ts.nextDouble();
                }
                for (int j = 0; j < out; j++) {
                    test_output_data[l][j] = ts.nextDouble();
                }
            }
        } catch (IOException e) {
            System.err.println("ERRO AO LER ARQUIVO DE TESTE: " + testPath);
            e.printStackTrace();
        }
    }

    public Base(int length) {
        this.length = length;
    }


    public void shuffle() {
        if (input_data.length == 0) {
            return;
        }

        double[] input_buffer = new double[in];
        double[] output_buffer = new double[out];

        Random rand = new Random();

        for (int i = length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);

            if (i == j) continue;

            System.arraycopy(input_data[i], 0, input_buffer, 0, input_data[i].length);
            System.arraycopy(output_data[i], 0, output_buffer, 0, output_data[i].length);

            System.arraycopy(input_data[j], 0, input_data[i], 0, input_data[i].length);
            System.arraycopy(output_data[j], 0, output_data[i], 0, output_data[i].length);

            System.arraycopy(input_buffer, 0, input_data[j], 0, input_data[i].length);
            System.arraycopy(output_buffer, 0, output_data[j], 0, output_data[i].length);
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int l = 0; l < test_length; l++) {
            for (int j = 0; j < in; j++) {
                sb.append((int) test_input_data[l][j]);
                sb.append(" ");
            }
            for (int j = 0; j < out; j++) {
                sb.append((int) test_output_data[l][j] == 1 ? 0.995 : 0.005);
                sb.append(" ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public void normalize() {
        if (in == 0 || length == 0) {
            return;
        }

        double[] minValues = new double[in];
        double[] maxValues = new double[in];

        for (int j = 0; j < in; j++) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < length; i++) {
                if (input_data[i][j] < min) {
                    min = input_data[i][j];
                }
                if (input_data[i][j] > max) {
                    max = input_data[i][j];
                }
            }
            minValues[j] = min;
            maxValues[j] = max;
        }

        for (int i = 0; i < length; i++) {
            for (int j = 0; j < in; j++) {
                double range = maxValues[j] - minValues[j];
                if (range == 0) {
                    input_data[i][j] = 0;
                } else {
                    input_data[i][j] = (input_data[i][j] - minValues[j]) / range;
                }
            }
        }

        for (int i = 0; i < test_length; i++) {
            for (int j = 0; j < in; j++) {
                double range = maxValues[j] - minValues[j];
                if (range == 0) {
                    test_input_data[i][j] = 0;
                } else {
                    test_input_data[i][j] = (test_input_data[i][j] - minValues[j]) / range;
                }
            }
        }
    }

    public int getNumTrainSamples() {
        return numTrainSamples;
    }

    public int getNumTestSamples() {
        return numTestSamples;
    }

    public int getIn() {
        return in;
    }

    public int getOut() {
        return out;
    }

    public int getHidden() {
        return hidden;
    }

    public double[][] getInput_data() {
        return input_data;
    }

    public double[][] getOutput_data() {
        return output_data;
    }

    public double[][] getTest_output_data() {
        return test_output_data;
    }

    public double[][] getTest_input_data() {
        return test_input_data;
    }
}
