package neural_networks;

import neural_networks.activations.ActivationFunction;
import neural_networks.activations.ReLU;
import neural_networks.activations.Sigmoid;
import neural_networks.core.MultilayerPerceptron;
import neural_networks.core.NeuralNetwork;
import neural_networks.utils.Base;
import neural_networks.utils.FileUtils;

import java.util.ArrayList;

public class Main {

    public static void main(String[] args) {
//        FileUtils fs = new FileUtils();
//        fs.splitFileByLastNumber("bases/datasets/abalone/adapted","bases/datasets/abalone");
//        fs.classIntoNeurons("abalone");
        Base base = new Base("tic-tac-toe");
        base.shuffle();
        base.normalize();
        MultilayerPerceptron rna = new MultilayerPerceptron(base, 10000, 0.3, 0.3);
        rna.train();
        rna.printLog(); //SOUT NOS RESULTADOS
        rna.generateCSV(); //GERA UM CSV PARA GERAÇÃO DE GRÁFICOS E RELATÓRIOS


        /*
        A PARTIR DAQUI É UMA IMPLEMENTAÇÃO BEM MAIS ROBUSTA DO PERCEPTRON MULTILAYER, USANDO FUNÇÕES PERSONALIZADAS PARA CADA CAMADA
        E USANDO UM VETOR "ARQUITETURA" PARA DEFINIR OS TAMANHSO DE CADA CAMADA
        */
        int[] architecture = {
                9,
                7,
                7,
                2
        }; //DEFINI QUANTAS CAMADAS, NO CASO, 9 DE ENTRADA, DUAS INTERMEDIARIAS COM 7 NEURONIOS CADA, E 2 DE SAIDA

        ArrayList<ActivationFunction> functions = new ArrayList<>();
        functions.add(new ReLU());
        functions.add(new ReLU());
        functions.add(new Sigmoid());
        //DEFINI UMA FUNÇÃO PRA CADA CAMADA, NO CASO A PRIMEIRA NÃO TEM FUNÇÃO.

        int epochs = 10000; //NUMERO DE EPOCAS

        double learningRate = 0.03; //TAXA DE APRENDIZADO (NI)

        double randomRange = 0.3; //RANGE DE VALORES QUE SERÁ UTILIZADO PARA INICIALIZAÇÃO (-0.3 ATÉ 0.3 )

//        NeuralNetwork nn = new NeuralNetwork(base, architecture, functions, epochs, learningRate, randomRange);
//        nn.train();
//        nn.printLog();
//        nn.generateCSV();

    }
}

