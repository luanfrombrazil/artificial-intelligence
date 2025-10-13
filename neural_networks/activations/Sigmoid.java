package neural_networks.activations;

public class Sigmoid implements ActivationFunction {
    @Override
    public double forward(double u) {
        return 1 / (1 + Math.exp(-u));
    }

    @Override
    public double derivative(double output) {
        return output * (1 - output);
    }
}