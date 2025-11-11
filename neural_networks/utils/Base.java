package neural_networks.utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
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

    private String name;
    private Random random;

    public String getName() {
        return name;
    }

    public Base(String name, int classes) {
        this.random = new Random();
        String testPath = "bases/datasets/" + name + "/" + name + ".test";
        try (Scanner ts = new Scanner(new File(testPath))) {
            this.test_length = ts.nextInt();
            this.in = ts.nextInt();
            this.out = ts.nextInt();
            this.hidden = (in + out) / 2;

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

        this.name = name.toUpperCase();
        String classesPath = "bases/datasets/" + name + "/classes/" + "class";


        //DESCOBRIR A MAIOR CLASSE
        int maior_classe = 0;
        for (int i = 1; i <= classes; i++) {
            try (Scanner sc = new Scanner(new File(classesPath + i))) {
                int tamanho = sc.nextInt();
                if (tamanho > maior_classe)
                    maior_classe = tamanho;
            } catch (IOException e) {
                System.err.println("ERRO AO LER ARQUIVO DE TREINO: " + classesPath + i);
                e.printStackTrace();
            }
        }
        ArrayList<double[]> input = new ArrayList<>();
        ArrayList<double[]> output = new ArrayList<>();
        for (int i = 1; i <= classes; i++) {
            ArrayList<double[]> linhas_input = new ArrayList<>();
            ArrayList<double[]> linhas_output = new ArrayList<>();
            String filename = classesPath + i;
            try (Scanner sc = new Scanner(new File(filename))) {
                int classsize = sc.nextInt();
                for (int j = 0; j < classsize; j++) {
                    double[] linhaIn = new double[in];
                    double[] linhaOut = new double[out];

                    for (int k = 0; k < in; k++) {
                        linhaIn[k] = sc.nextDouble();
                    }
                    sc.nextDouble();//FLUSH NO ULTIMO NUMERO

                    linhaOut[linhaOut.length - i] = 1;
                    linhas_output.add(linhaOut);
                    linhas_input.add(linhaIn);
                }

                while (linhas_input.size() < maior_classe) {
                    linhas_input.add(randomize(linhas_input));
                    linhas_output.add(duplicate(linhas_output));
                }
            } catch (IOException e) {
                System.err.println("ERRO AO LER ARQUIVO DE TREINO: " + classesPath + i);
                e.printStackTrace();
            }
            input.addAll(linhas_input);
            output.addAll(linhas_output);
        }
        this.input_data = input.toArray(new double[0][0]);
        this.output_data = output.toArray(new double[0][0]);
        this.length = input_data.length;
    }

    public double[] randomize(ArrayList<double[]> lista) {
        double[] sample = lista.get(this.random.nextInt(lista.size()));
        //RANDOMIZAR A SAMPLE AQUI
        return sample;
    }

    public double[] duplicate(ArrayList<double[]> lista) {
        double[] sample = lista.get(this.random.nextInt(lista.size()));
        return sample;
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