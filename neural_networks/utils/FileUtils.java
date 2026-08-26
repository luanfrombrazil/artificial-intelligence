package neural_networks.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.Scanner;
import java.util.stream.Stream;

/**
 * Utilitários de preparação das bases de dados.
 *
 * <p>Estas rotinas transformam um arquivo bruto — uma linha por amostra, com o rótulo
 * numérico na última coluna — nos arquivos particionados por classe que
 * {@link Base} espera encontrar em {@code bases/datasets/<nome>/classes/}.</p>
 *
 * <p>São de uso pontual: rodam uma vez, quando a base é preparada, e não fazem parte do
 * ciclo de treinamento. Por isso ficam comentadas no {@code Main}.</p>
 *
 * <p><b>Atenção antes de reexecutar.</b> Ambos os métodos abrem os arquivos de saída em
 * modo de <em>acréscimo</em>. Rodar a preparação duas vezes sobre a mesma pasta duplica
 * silenciosamente todo o conteúdo, sem erro nem aviso. Apague os arquivos gerados antes
 * de repetir a operação.</p>
 */
public class FileUtils {

    /** Localidade dos arquivos de dados, que usam vírgula como separador decimal. */
    private static final Locale LOCALE_DADOS = Locale.forLanguageTag("pt-BR");

    /**
     * Converte arquivos de classe com rótulo numérico em arquivos com saída codificada
     * em um-de-N.
     *
     * <p>Lê cada {@code classN}, descarta o rótulo bruto e escreve, em {@code classN-n},
     * os atributos seguidos de um vetor com 1 na posição da classe e 0 nas demais.</p>
     *
     * @param base           nome da base, correspondente à pasta em {@code bases/datasets/}
     * @param nClasses       número de arquivos de classe a percorrer, de 1 a {@code nClasses}
     * @param nAtributos     número de atributos de entrada por linha
     * @param classesAPular  rótulos a ignorar, para bases em que algum valor não ocorre.
     *                       Pode ser vazio
     */
    public void classIntoNeurons(String base, int nClasses, int nAtributos, int... classesAPular) {
        for (int i = 1; i <= nClasses; i++) {
            if (devePular(i, classesAPular)) {
                continue;
            }
            String inputFilePath = "bases/datasets/" + base + "/class" + i;
            try (Scanner sc = new Scanner(new File(inputFilePath)).useLocale(LOCALE_DADOS);
                 BufferedWriter bw = new BufferedWriter(new FileWriter(inputFilePath + "-n", true))) {

                while (sc.hasNext()) {
                    StringBuilder sb = new StringBuilder();
                    for (int j = 0; j < nAtributos; j++) {
                        sb.append(sc.nextDouble()).append(" ");
                    }
                    sc.nextDouble(); // consome o rotulo bruto
                    for (int j = 1; j <= nClasses; j++) {
                        sb.append(j == i ? "1 " : "0 ");
                    }
                    bw.write(sb.toString());
                    bw.newLine();
                }
                bw.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Verifica se um rótulo está na lista de exclusões.
     *
     * @param classe        rótulo a testar
     * @param classesAPular rótulos a ignorar
     * @return {@code true} se o rótulo deve ser pulado
     */
    private boolean devePular(int classe, int... classesAPular) {
        for (int pular : classesAPular) {
            if (classe == pular) {
                return true;
            }
        }
        return false;
    }

    /**
     * Particiona um arquivo de amostras em vários arquivos, um por valor de rótulo.
     *
     * <p>O rótulo é lido do último campo de cada linha, e a linha inteira é acrescentada
     * ao arquivo {@code class<rotulo>} do diretório de saída. Linhas em branco são
     * ignoradas.</p>
     *
     * @param inputFilePath   caminho do arquivo de origem, uma amostra por linha, com o
     *                        rótulo no último campo
     * @param outputDirectory diretório onde os arquivos {@code classN} serão criados;
     *                        é criado se não existir
     */
    public void splitFileByLastNumber(String inputFilePath, String outputDirectory) {
        try {
            Files.createDirectories(Paths.get(outputDirectory));
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        try (Stream<String> lines = Files.lines(Paths.get(inputFilePath))) {
            lines.forEach(line -> {
                String trimmedLine = line.trim();
                if (trimmedLine.isEmpty()) {
                    return;
                }
                String[] parts = trimmedLine.split("\\s+");
                String classificationKey = parts[parts.length - 1];
                Path outputFilePath = Paths.get(outputDirectory, "class" + classificationKey);

                try {
                    Files.writeString(outputFilePath, line + System.lineSeparator(),
                            StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
