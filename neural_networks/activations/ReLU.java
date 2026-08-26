package neural_networks.activations;

/**
 * Função de ativação ReLU (<em>Rectified Linear Unit</em>).
 *
 * <pre>
 *   f(u)  = max(0, u)               faixa [0, +infinito)
 *   f'(u) = 1 se u &gt; 0, senao 0
 * </pre>
 *
 * <p>É a ativação que destravou as redes profundas. Enquanto a sigmoide tem derivada
 * máxima de 0,25 — e portanto encolhe o gradiente em pelo menos quatro vezes a cada
 * camada atravessada —, a ReLU tem derivada exatamente 1 em toda a região ativa. O
 * gradiente atravessa muitas camadas sem se dissolver, e o cálculo é uma única
 * comparação.</p>
 *
 * <p><b>Ponto fraco: o neurônio morto.</b> Se um neurônio passa a receber entrada
 * líquida negativa para todas as amostras, sua saída é sempre 0, sua derivada é sempre 0
 * e nenhum gradiente volta a alcançá-lo — ele nunca mais se recupera. Uma taxa de
 * aprendizado alta ou uma inicialização mal dimensionada podem matar boa parte da
 * camada logo nas primeiras épocas, e a rede estabiliza num estado em que a saída é
 * constante. Inicialização de He e taxas moderadas são as defesas usuais.</p>
 */
public class ReLU implements ActivationFunction {

    /**
     * Retifica a entrada líquida.
     *
     * @param u entrada líquida do neurônio
     * @return {@code u} se positivo, {@code 0} caso contrário
     */
    @Override
    public double forward(double u) {
        return Math.max(0, u);
    }

    /**
     * Calcula a derivada da ReLU a partir da saída já conhecida.
     *
     * <p>Na origem a função não é diferenciável; por convenção adota-se derivada 0,
     * o que na prática não causa problema porque o caso tem medida nula.</p>
     *
     * @param output saída previamente calculada por {@link #forward(double)}
     * @return {@code 1} se a saída é positiva, {@code 0} caso contrário
     */
    @Override
    public double derivative(double output) {
        return output > 0 ? 1 : 0;
    }
}
