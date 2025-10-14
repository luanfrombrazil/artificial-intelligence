package neural_networks.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Scanner;
import java.util.stream.Stream;

public class FileUtils {

    public void classIntoNeurons(String base) {
        for (int i = 1; i <= 29; i++) {
            if (i == 28)
                continue;
            String inputFilePath = "bases/datasets/" + base + "/class" + i;
            try (Scanner sc = new Scanner(new File(inputFilePath))) {
                BufferedWriter bw = new BufferedWriter(new FileWriter(inputFilePath + "-n", true));
                while (sc.hasNext()) {
                    StringBuilder sb = new StringBuilder();
                    for (int j = 0; j < 10; j++) {
                        sb.append(sc.nextDouble() + "");
                        sb.append(" ");
                    }
                    sc.nextDouble();
                    for (int j = 1; j <= 29; j++) {
                        if (j == i)
                            sb.append("1 ");
                        else
                            sb.append("0 ");
                    }
                    String test = sb.toString();
//                    System.out.println(test);

                    bw.write(test);
                    bw.newLine();
                    bw.flush();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void splitFileByLastNumber(String inputFilePath, String outputDirectory) {
        for (int j = 1; j <= 29; j++) {
            try {
                Files.createDirectories(Paths.get(outputDirectory));
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }
        }
        try (Stream<String> lines = Files.lines(Paths.get(inputFilePath))) {
            lines.forEach(line -> {
                String trimmedLine = line.trim();
                if (trimmedLine.isEmpty()) {
                    return;
                }
                String[] parts = trimmedLine.split("\\s+");
                if (parts.length > 0) {
                    String classificationKey = parts[parts.length - 1];
                    String outputFileName = "class" + classificationKey;
                    Path outputFilePath = Paths.get(outputDirectory, outputFileName);

                    try {
                        Files.writeString(outputFilePath, line + System.lineSeparator(),
                                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

//    public void appendAtTheEnd(String base) {
//            String inputFilePath = "bases/datasets/" + base;
//            try (Scanner sc = new Scanner(new File(inputFilePath))) {
//                BufferedWriter bw = new BufferedWriter(new FileWriter(inputFilePath + "-n", true));
//                while (sc.hasNext()) {
//                    StringBuilder sb = new StringBuilder();
//                    for (int j = 0; j < 10; j++) {
//                        sb.append(sc.nextDouble() + "");
//                        sb.append(" ");
//                    }
//                    sc.nextDouble();
//                    for (int j = 1; j <= 29; j++) {
//                        if (j == i)
//                            sb.append("1 ");
//                        else
//                            sb.append("0 ");
//                    }
//                    String test = sb.toString();
////                    System.out.println(test);
//
//                    bw.write(test);
//                    bw.newLine();
//                    bw.flush();
//                }
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }
//    }
}