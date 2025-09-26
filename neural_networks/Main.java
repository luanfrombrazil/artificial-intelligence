package neural_networks;

public class Main {

    public static void main(String[] args) {
        Base base = new Base("and");Perceptron simplerna = new Perceptron(base, 1000, 0.3, 0.3);
        simplerna.train(true);
        simplerna.status(true,false);

        MultilayerPerceptron rna = new MultilayerPerceptron(base, 1000, 0.3, 0.3);
        rna.train(true);
        rna.status(true, false);

    }
}

