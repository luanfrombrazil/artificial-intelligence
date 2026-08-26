package neural_networks.activations;

/**
 * Contrato de uma função de ativação.
 *
 * <p>A função de ativação é o que dá à rede neural o seu poder. Sem ela, cada neurônio
 * calcularia apenas uma combinação linear das entradas — e a composição de camadas
 * lineares é ela própria uma função linear. Uma rede de dez camadas sem ativação não
 * consegue representar nada que um único neurônio já não representasse, e continuaria
 * incapaz de resolver o XOR. É a não-linearidade introduzida aqui que permite à rede
 * dobrar o espaço de entrada e separar classes que não são linearmente separáveis.</p>
 *
 * <p>Cada implementação precisa fornecer duas coisas: o valor da função (usado no passo
 * para a frente) e a sua derivada (usada no passo para trás, pela regra da cadeia do
 * algoritmo de retropropagação).</p>
 */
public interface ActivationFunction {

    /**
     * Aplica a função de ativação à entrada líquida de um neurônio.
     *
     * @param u soma ponderada das entradas mais o viés (o {@code net} do neurônio)
     * @return o valor de saída do neurônio
     */
    double forward(double u);

    /**
     * Devolve a derivada da função de ativação no ponto correspondente a uma saída já
     * calculada.
     *
     * <p><b>Atenção à convenção:</b> o parâmetro é a <em>saída</em> do neurônio, não a
     * entrada líquida. Isso é possível porque, nas funções usadas aqui, a derivada se
     * escreve em função do próprio valor de saída — a sigmoide tem
     * {@code f'(u) = f(u)(1 - f(u))}, e a ReLU tem derivada 1 onde a saída é positiva.
     * A vantagem é não precisar guardar {@code u}; a limitação é que funções cuja
     * derivada não se expressa assim (softmax, por exemplo) não cabem nesta interface
     * sem alteração.</p>
     *
     * @param output valor previamente devolvido por {@link #forward(double)} para o
     *               mesmo neurônio
     * @return a derivada da função de ativação naquele ponto
     */
    double derivative(double output);
}
