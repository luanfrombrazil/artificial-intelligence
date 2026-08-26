package neural_networks.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Coleta e exporta as métricas de um treinamento, época a época.
 *
 * <p>Acompanha quatro séries por conjunto — erro médio e número de classificações
 * incorretas, em treino, validação e teste — e as materializa em dois formatos: um
 * relatório legível em {@code .log} e um {@code .csv} para geração de gráficos.</p>
 *
 * <p>As três séries juntas são o que permite diagnosticar um treinamento em vez de apenas
 * observá-lo. Erro de treino caindo enquanto o de validação sobe é a assinatura clássica
 * do sobreajuste; as duas séries paradas desde a primeira época indicam rede saturada;
 * validação bem abaixo do teste sugere que os dois conjuntos não vêm da mesma
 * distribuição.</p>
 *
 * <p><b>Nota sobre nomenclatura.</b> O erro registrado pelas classes deste projeto é a
 * soma dos desvios absolutos por amostra, não o erro quadrático médio. As colunas usam
 * o termo neutro "erro" justamente para não repetir a designação incorreta de MSE que
 * existia antes.</p>
 */
public class Statistics {

    /** Nome da base, usado nos nomes dos arquivos gerados. */
    private final String baseName;

    /** Erro médio de treino por época. */
    private final List<Double> trainErrorHistory = new ArrayList<>();

    /** Erro médio de validação por época; {@code NaN} quando não há validação. */
    private final List<Double> validationErrorHistory = new ArrayList<>();

    /** Erro médio de teste por época. */
    private final List<Double> testErrorHistory = new ArrayList<>();

    /** Classificações incorretas no treino por época. */
    private final List<Integer> trainClassificationErrorHistory = new ArrayList<>();

    /** Classificações incorretas na validação por época. */
    private final List<Integer> validationClassificationErrorHistory = new ArrayList<>();

    /** Classificações incorretas no teste por época. */
    private final List<Integer> testClassificationErrorHistory = new ArrayList<>();

    /** Época escolhida pela parada antecipada (1-based); -1 se não houve. */
    private int melhorEpocaValidacao = -1;

    /** Intervalo de épocas entre impressões de progresso na saída padrão. */
    private int intervaloImpressao = 50;

    /**
     * Cria um coletor de métricas para uma base.
     *
     * @param baseName nome da base, usado para nomear os arquivos de saída
     */
    public Statistics(String baseName) {
        this.baseName = baseName;
    }

    /**
     * Define de quantas em quantas épocas o progresso é impresso na saída padrão.
     *
     * @param intervalo número de épocas entre impressões; valores menores ou iguais a
     *                  zero silenciam a saída
     */
    public void setIntervaloImpressao(int intervalo) {
        this.intervaloImpressao = intervalo;
    }

    /**
     * Registra as métricas de uma época com conjunto de validação.
     *
     * @param trainError                   erro médio no conjunto de treino
     * @param trainClassificationErrors    amostras de treino classificadas incorretamente
     * @param validationError              erro médio no conjunto de validação, ou
     *                                     {@link Double#NaN} se não houver validação
     * @param validationClassificationErrors amostras de validação classificadas incorretamente
     * @param testError                    erro médio no conjunto de teste
     * @param testClassificationErrors     amostras de teste classificadas incorretamente
     */
    public void recordEpoch(double trainError, int trainClassificationErrors,
                            double validationError, int validationClassificationErrors,
                            double testError, int testClassificationErrors) {
        trainErrorHistory.add(trainError);
        trainClassificationErrorHistory.add(trainClassificationErrors);
        validationErrorHistory.add(validationError);
        validationClassificationErrorHistory.add(validationClassificationErrors);
        testErrorHistory.add(testError);
        testClassificationErrorHistory.add(testClassificationErrors);

        int epoch = trainErrorHistory.size();

        if (intervaloImpressao > 0 && epoch % intervaloImpressao == 0) {
            if (Double.isNaN(validationError)) {
                System.out.printf(Locale.US,
                        "Epoca: %-6d | Erros Treino: %-6d | Erros Teste: %-6d | Erro Treino: %.6f | Erro Teste: %.6f%n",
                        epoch, trainClassificationErrors, testClassificationErrors, trainError, testError);
            } else {
                System.out.printf(Locale.US,
                        "Epoca: %-6d | Erros Treino: %-6d | Erros Valid.: %-6d | Erros Teste: %-6d | "
                        + "Erro Treino: %.6f | Erro Valid.: %.6f | Erro Teste: %.6f%n",
                        epoch, trainClassificationErrors, validationClassificationErrors, testClassificationErrors,
                        trainError, validationError, testError);
            }
        }
    }

    /**
     * Registra as métricas de uma época sem conjunto de validação.
     *
     * <p>Mantido para {@link neural_networks.core.Perceptron} e
     * {@link neural_networks.core.MultilayerPerceptron}, que não implementam validação.</p>
     *
     * @param trainError                erro médio no conjunto de treino
     * @param trainClassificationErrors amostras de treino classificadas incorretamente
     * @param testError                 erro médio no conjunto de teste
     * @param testClassificationErrors  amostras de teste classificadas incorretamente
     */
    public void recordEpoch(double trainError, int trainClassificationErrors,
                            double testError, int testClassificationErrors) {
        recordEpoch(trainError, trainClassificationErrors, Double.NaN, 0, testError, testClassificationErrors);
    }

    /**
     * Informa qual época foi escolhida pela parada antecipada.
     *
     * @param epoca época de melhor erro de validação, 1-based
     */
    public void setMelhorEpocaValidacao(int epoca) {
        this.melhorEpocaValidacao = epoca;
    }

    /**
     * Escreve o relatório detalhado em {@code data/[timestamp] BASE.log}.
     *
     * @param trainSize      número de amostras de treino, usado para converter erros em acurácia
     * @param validationSize número de amostras de validação; zero se não houver
     * @param testSize       número de amostras de teste
     */
    public void writeDetailedLog(int trainSize, int validationSize, int testSize) {
        String filename = nomeDeArquivo("log");
        garantirDiretorio(filename);

        boolean temValidacao = validationSize > 0;

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
            int firstPerfectTrain = findFirstEpochWithZeroErrors(trainClassificationErrorHistory);
            int firstPerfectTest = findFirstEpochWithZeroErrors(testClassificationErrorHistory);
            int bestValidationEpoch = findBestEpoch(validationClassificationErrorHistory);
            int lastTrainErrors = ultimoValor(trainClassificationErrorHistory);
            int lastTestErrors = ultimoValor(testClassificationErrorHistory);

            bw.write("BASE: " + baseName + "\n");
            bw.write("EPOCAS EXECUTADAS: " + trainErrorHistory.size() + "\n");
            bw.write("AMOSTRAS: treino=" + trainSize + " validacao=" + validationSize + " teste=" + testSize + "\n");
            bw.write("PRIMEIRA EPOCA TREINO SEM ERRO: " + (firstPerfectTrain == -1 ? "NAO CONVERGIU" : firstPerfectTrain) + "\n");
            bw.write("PRIMEIRA EPOCA TESTE SEM ERRO: " + (firstPerfectTest == -1 ? "NAO CONVERGIU" : firstPerfectTest) + "\n");
            if (temValidacao) {
                bw.write("MELHOR EPOCA POR VALIDACAO: " + (bestValidationEpoch == -1 ? "N/A" : bestValidationEpoch) + "\n");
                bw.write("EPOCA ESCOLHIDA PELA PARADA ANTECIPADA: "
                        + (melhorEpocaValidacao == -1 ? "N/A" : melhorEpocaValidacao) + "\n");
            } else {
                bw.write("SEM CONJUNTO DE VALIDACAO: a selecao de epoca nao pode ser feita sem vazar o teste.\n");
            }
            bw.write("CLASSIFICOU TREINO CORRETAMENTE AO FINAL: " + (lastTrainErrors == 0 ? "SIM" : "NAO") + "\n");
            bw.write("CLASSIFICOU TESTE CORRETAMENTE AO FINAL: " + (lastTestErrors == 0 ? "SIM" : "NAO") + "\n");
            bw.write("\n\n");

            String cabecalho = temValidacao
                    ? String.format("%-7s | %-12s | %-12s | %-10s | %-12s | %-12s | %-10s | %-11s | %-12s | %-10s |%n",
                            "EPOCA", "ERROS TREINO", "ERRO TREINO", "ACC TREINO", "ERROS VALID", "ERRO VALID",
                            "ACC VALID", "ERROS TESTE", "ERRO TESTE", "ACC TESTE")
                    : String.format("%-7s | %-12s | %-12s | %-10s | %-11s | %-12s | %-10s |%n",
                            "EPOCA", "ERROS TREINO", "ERRO TREINO", "ACC TREINO", "ERROS TESTE", "ERRO TESTE", "ACC TESTE");

            for (int i = 0; i < trainErrorHistory.size(); i++) {
                if (i % 100 == 0) {
                    bw.write("===================================== RELATORIO DE TREINAMENTO =====================================\n");
                    bw.write(cabecalho);
                    bw.write("---------------------------------------------------------------------------------------------------\n");
                }

                int trainErrors = trainClassificationErrorHistory.get(i);
                double trainAccuracy = ((double) trainSize - trainErrors) / trainSize;
                int testErrors = testClassificationErrorHistory.get(i);
                double testAccuracy = ((double) testSize - testErrors) / testSize;

                if (temValidacao) {
                    int valErrors = validationClassificationErrorHistory.get(i);
                    double valAccuracy = ((double) validationSize - valErrors) / validationSize;
                    bw.write(String.format(Locale.US,
                            "%-7d | %-12d | %-12.8f | %-9.1f%% | %-12d | %-12.8f | %-9.1f%% | %-11d | %-12.8f | %-9.1f%% |%n",
                            (i + 1), trainErrors, trainErrorHistory.get(i), trainAccuracy * 100,
                            valErrors, validationErrorHistory.get(i), valAccuracy * 100,
                            testErrors, testErrorHistory.get(i), testAccuracy * 100));
                } else {
                    bw.write(String.format(Locale.US,
                            "%-7d | %-12d | %-12.8f | %-9.1f%% | %-11d | %-12.8f | %-9.1f%% |%n",
                            (i + 1), trainErrors, trainErrorHistory.get(i), trainAccuracy * 100,
                            testErrors, testErrorHistory.get(i), testAccuracy * 100));
                }
            }
            bw.write("===================================================================================================\n");

        } catch (IOException e) {
            System.err.println("Erro ao escrever log detalhado: " + filename);
            e.printStackTrace();
        }
    }

    /**
     * Escreve o relatório detalhado sem conjunto de validação.
     *
     * @param trainSize número de amostras de treino
     * @param testSize  número de amostras de teste
     */
    public void writeDetailedLog(int trainSize, int testSize) {
        writeDetailedLog(trainSize, 0, testSize);
    }

    /**
     * Exporta o histórico por época em CSV, com ponto decimal e vírgula como separador
     * de campos, pronto para ser lido por {@code plotCharts.py}.
     */
    public void exportToCSV() {
        String filename = nomeDeArquivo("csv");
        garantirDiretorio(filename);

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
            bw.write("epoch,train_error,train_classification_errors,"
                    + "validation_error,validation_classification_errors,"
                    + "test_error,test_classification_errors\n");

            for (int i = 0; i < trainErrorHistory.size(); i++) {
                double val = validationErrorHistory.get(i);
                bw.write(String.format(Locale.US, "%d,%.8f,%d,%s,%d,%.8f,%d%n",
                        i + 1,
                        trainErrorHistory.get(i),
                        trainClassificationErrorHistory.get(i),
                        Double.isNaN(val) ? "" : String.format(Locale.US, "%.8f", val),
                        validationClassificationErrorHistory.get(i),
                        testErrorHistory.get(i),
                        testClassificationErrorHistory.get(i)));
            }
        } catch (IOException e) {
            System.err.println("Erro ao exportar estatisticas para CSV: " + filename);
            e.printStackTrace();
        }
    }

    /**
     * Imprime um resumo curto do treinamento na saída padrão.
     */
    public void printFinalSummary() {
        System.out.println("====== RESUMO DO TREINAMENTO ======");
        System.out.println("Base: " + baseName);
        System.out.println("Total de Epocas: " + trainErrorHistory.size());
        int melhor = findBestEpoch(validationClassificationErrorHistory);
        if (melhor != -1) {
            System.out.printf("Melhor epoca por validacao: %d (com %d erros)%n",
                    melhor, validationClassificationErrorHistory.get(melhor - 1));
        }
        System.out.println("===================================");
    }

    /**
     * Monta o nome do arquivo de saída, com carimbo de tempo em milissegundos.
     *
     * <p>A resolução em milissegundos elimina a necessidade da pausa artificial que
     * existia entre duas execuções seguidas apenas para evitar colisão de nomes.</p>
     *
     * @param extensao extensão do arquivo, sem o ponto
     * @return caminho relativo dentro de {@code data/}
     */
    private String nomeDeArquivo(String extensao) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss-SSS");
        String timestamp = LocalDateTime.now().format(formatter);
        return String.format("data/[%s] %s.%s", timestamp, baseName, extensao);
    }

    /**
     * Cria o diretório do arquivo caso ainda não exista, evitando que
     * {@link FileWriter} falhe por diretório ausente.
     *
     * @param caminho caminho do arquivo a ser escrito
     */
    private void garantirDiretorio(String caminho) {
        File pai = new File(caminho).getAbsoluteFile().getParentFile();
        if (pai != null && !pai.exists() && !pai.mkdirs()) {
            System.err.println("Nao foi possivel criar o diretorio: " + pai);
        }
    }

    /**
     * Devolve o último valor de um histórico.
     *
     * @param historico série a consultar
     * @return o último valor, ou -1 se a série estiver vazia
     */
    private int ultimoValor(List<Integer> historico) {
        return historico.isEmpty() ? -1 : historico.get(historico.size() - 1);
    }

    /**
     * Localiza a primeira época com zero classificações incorretas.
     *
     * @param errorHistory série de erros de classificação por época
     * @return a época (1-based), ou -1 se nunca ocorreu
     */
    private int findFirstEpochWithZeroErrors(List<Integer> errorHistory) {
        for (int i = 0; i < errorHistory.size(); i++) {
            if (errorHistory.get(i) == 0) {
                return i + 1;
            }
        }
        return -1;
    }

    /**
     * Localiza a época de menor número de classificações incorretas.
     *
     * @param errorHistory série de erros de classificação por época
     * @return a época (1-based), ou -1 se a série estiver vazia ou toda zerada por
     *         ausência do conjunto correspondente
     */
    private int findBestEpoch(List<Integer> errorHistory) {
        if (errorHistory.isEmpty()) {
            return -1;
        }
        int bestEpoch = 1;
        int minErrors = errorHistory.get(0);
        boolean algumPositivo = minErrors > 0;
        for (int i = 1; i < errorHistory.size(); i++) {
            if (errorHistory.get(i) > 0) {
                algumPositivo = true;
            }
            if (errorHistory.get(i) < minErrors) {
                minErrors = errorHistory.get(i);
                bestEpoch = i + 1;
            }
        }
        return algumPositivo ? bestEpoch : -1;
    }
}
