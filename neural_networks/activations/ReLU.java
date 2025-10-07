package neural_networks.activations;

public class ReLU implements ActivationFunction {
    @Override
    public double forward(double u) {
        return Math.max(0, u);
    }

    @Override
    public double derivative(double output) {
        return output > 0 ? 1 : 0;
    }
}