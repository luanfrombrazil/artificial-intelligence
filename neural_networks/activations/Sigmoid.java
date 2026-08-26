package neural_networks.activations;

/**
 * Função de ativação sigmoide logística.
 *
 * <pre>
 *   f(u)  = 1 / (1 + e^-u)          faixa (0, 1)
 *   f'(u) = f(u) * (1 - f(u))
 * </pre>
 *
 * <p>Foi a ativação que tornou a retropropagação possível. O perceptron original de
 * Rosenblatt usava um degrau, cuja derivada é zero em toda parte e indefinida na origem
 * — não há gradiente para propagar. A sigmoide é a versão suave e diferenciável do
 * degrau: mantém a interpretação de "quanto o neurônio está ativado", entre 0 e 1, e
 * ainda por cima tem derivada de forma fechada e barata, expressa na própria saída.</p>
 *
 * <p><b>Ponto fraco.</b> Para {@code |u|} grande a saída satura perto de 0 ou de 1 e a
 * derivada tende a zero. O gradiente que chega às camadas anteriores é multiplicado por
 * esses valores pequenos a cada camada e desaparece — é o problema do
 * <em>vanishing gradient</em>, a razão histórica pela qual redes profundas com sigmoide
 * eram difíceis de treinar e pela qual a ReLU passou a ser preferida nas camadas
 * ocultas.</p>
 */
public class Sigmoid implements ActivationFunction {

    /**
     * Calcula a sigmoide logística.
     *
     * @param u entrada líquida do neurônio
     * @return valor no intervalo aberto (0, 1)
     */
    @Override
    public double forward(double u) {
        return 1 / (1 + Math.exp(-u));
    }

    /**
     * Calcula a derivada da sigmoide a partir da saída já conhecida.
     *
     * @param output saída previamente calculada por {@link #forward(double)}
     * @return {@code output * (1 - output)}; máximo de 0,25 em {@code output = 0,5} e
     *         próximo de zero nas duas extremidades
     */
    @Override
    public double derivative(double output) {
        return output * (1 - output);
    }
}
