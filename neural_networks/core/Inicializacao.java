package neural_networks.core;

/**
 * Estratégias de inicialização dos pesos de uma camada.
 *
 * <p>Inicializar pesos parece um detalhe, mas é uma das decisões que mais determinam se
 * uma rede vai treinar ou morrer nas primeiras épocas. O problema é de escala: a entrada
 * líquida de um neurônio é a soma de {@code nInputs} produtos. Se os pesos forem grandes
 * demais, essa soma cresce com o número de entradas, a ativação satura e o gradiente
 * desaparece; se forem pequenos demais, o sinal se dissolve ao atravessar as camadas.</p>
 *
 * <p>As estratégias modernas resolvem isso dimensionando a variância dos pesos pelo
 * <em>fan-in</em> — o número de entradas da camada —, de modo que a variância do sinal
 * se mantenha aproximadamente constante ao longo da rede.</p>
 */
public enum Inicializacao {

    /**
     * Distribuição uniforme em {@code [-randomRange, +randomRange]}, ignorando o número
     * de entradas da camada.
     *
     * <p>É a estratégia original do projeto, mantida como padrão para preservar o
     * comportamento histórico. Funciona bem em redes rasas e com poucas entradas, mas
     * não acompanha o crescimento do fan-in: com dez entradas e ReLU, a combinação de
     * faixa fixa com taxa de aprendizado alta costuma levar ao colapso da camada.</p>
     */
    UNIFORME,

    /**
     * Xavier (ou Glorot): distribuição normal com desvio {@code sqrt(1 / nInputs)}.
     *
     * <p>Deduzida para ativações simétricas em torno de zero e aproximadamente lineares
     * perto da origem — sigmoide e tanh. Mantém a variância do sinal estável na ida e a
     * do gradiente na volta.</p>
     */
    XAVIER,

    /**
     * He (ou Kaiming): distribuição normal com desvio {@code sqrt(2 / nInputs)}.
     *
     * <p>Variante de Xavier corrigida para a ReLU. Como a ReLU zera metade das ativações,
     * ela também corta pela metade a variância que passa adiante; o fator 2 compensa
     * exatamente essa perda. É a escolha padrão para camadas com ReLU.</p>
     */
    HE
}
