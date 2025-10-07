package neural_networks;

import neural_networks.core.MultilayerPerceptron;
import neural_networks.utils.Base;

public class Main {

    public static void main(String[] args) {
        Base base = new Base("tic-tac-toe-int");
//        System.out.println(base);
//
        MultilayerPerceptron rna = new MultilayerPerceptron(base, 10000, 0.3, 0.3);
        rna.train();
        rna.printLog(true, true);
    }
}

