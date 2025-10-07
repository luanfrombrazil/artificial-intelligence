package neural_networks.core;

import neural_networks.activations.ActivationFunction;

import java.util.Random;

public class Layer {
    public final double[][] weights;
    public final double[] biases;
    private final ActivationFunction activationFunction;

    private double[] lastInputs;
    private double[] lastOutputs;

    public Layer(int nInputs, int nNeurons, ActivationFunction activationFunction, double range) {
        this.activationFunction = activationFunction;
        this.weights = new double[nInputs][nNeurons];
        this.biases = new double[nNeurons];

        Random random = new Random();
        for (int i = 0; i < nInputs; i++) {
            for (int j = 0; j < nNeurons; j++) {
                weights[i][j] = random.nextDouble(range * 2) - range;
            }
        }
        for (int j = 0; j < nNeurons; j++) {
            biases[j] = random.nextDouble(range * 2) - range;
        }
    }

    public double[] forward(double[] inputs) {
        this.lastInputs = inputs;
        int nNeurons = biases.length;
        double[] outputs = new double[nNeurons];

        for (int j = 0; j < nNeurons; j++) {
            double u = 0;
            for (int i = 0; i < inputs.length; i++) {
                u += inputs[i] * weights[i][j];
            }
            u += biases[j];
            outputs[j] = activationFunction.forward(u);
        }
        this.lastOutputs = outputs;
        return outputs;
    }

    public double[] getLastInputs() {
        return lastInputs;
    }

    public double[] getLastOutputs() {
        return lastOutputs;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}