package neural_networks.activations;

public interface ActivationFunction {
    /**
     * APLICA A FUNÇÃO DE ATIVAÇÃO
     * @param u SOMA PONDERADA.
     * @return VALOR DA SAÍDA APÓS ATIVAÇÃO.
     */
    double forward(double u);

    /**
     * A DERIVADA DA FUNÇÃO DE ATIVAÇÃO
     * @param output O VALOR DE SAIDA DA FUNÇÃO DE ATIVAÇÃO (RESULTADO DE forward(u)).
     * @return VALOR DA DERIVADA.
     */
    double derivative(double output);
}