package neural_networks.core;

import neural_networks.activations.ActivationFunction;

import java.util.Random;

public class Layer {
    private final double[][] weights;
    private final double[] biases;

    private final ActivationFunction activationFunction;

    private final double[][] weightGradients;
    private final double[] biasGradients;

    private double[] lastInputs;
    private double[] lastOutputs;

    public Layer(int nInputs, int nNeurons, ActivationFunction activationFunction, double randomRange, boolean isLastLayer) {

        this.activationFunction = activationFunction;
        this.weights = new double[nInputs][nNeurons];
        this.biases = new double[nNeurons];

        Random random = new Random();
        for (int i = 0; i < nInputs; i++) {
            for (int j = 0; j < nNeurons; j++) {
                weights[i][j] = random.nextDouble(randomRange * 2) - randomRange;
            }
        }
        for (int j = 0; j < nNeurons; j++) {
            biases[j] = random.nextDouble(randomRange * 2) - randomRange;
        }

        this.weightGradients = new double[nInputs][nNeurons];
        this.biasGradients = new double[nNeurons];
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
            outputs[j] = this.activationFunction.forward(u);
        }

        this.lastOutputs = outputs;
        return outputs;
    }

    public double[] backward(double[] deltaFromNextLayer) {
        int nInputs = lastInputs.length;
        int nNeurons = biases.length;
        double[] deltaForPreviousLayer = new double[nInputs]; // Erro a ser passado para trás

        for (int j = 0; j < nNeurons; j++) {
            double derivative = this.activationFunction.derivative(this.lastOutputs[j]);
            double delta = deltaFromNextLayer[j] * derivative;

            biasGradients[j] = delta;
            for (int i = 0; i < nInputs; i++) {
                weightGradients[i][j] = lastInputs[i] * delta;
                deltaForPreviousLayer[i] += weights[i][j] * delta;
            }
        }

        return deltaForPreviousLayer;
    }

    public void updateParameters(double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] += learningRate * weightGradients[i][j];
            }
        }
        for (int j = 0; j < biases.length; j++) {
            biases[j] += learningRate * biasGradients[j];
        }
    }
}

