package neural_networks.activations;

public interface ActivationFunction {

    double forward(double u);

    double derivative(double output);
}