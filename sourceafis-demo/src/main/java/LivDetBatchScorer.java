import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

public class LivDetBatchScorer {
    private static List<Path> collectPngs(Path dir, String globPattern) throws IOException {
        if (!Files.isDirectory(dir)) {
            throw new RuntimeException("Directory does not exist: " + dir);
        }

        List<Path> files = new ArrayList<>();
        PathMatcher matcher = FileSystems.getDefault().getPathMatcher("glob:" + globPattern);

        try (Stream<Path> walk = Files.walk(dir)) {
            walk.filter(Files::isRegularFile)
                .filter(p -> p.getFileName().toString().toLowerCase().endsWith(".png"))
                .filter(p -> matcher.matches(p.getFileName()))
                .sorted()
                .forEach(files::add);
        }

        return files;
    }

    private static List<Path> collectAllPngs(Path dir) throws IOException {
        if (!Files.isDirectory(dir)) {
            throw new RuntimeException("Directory does not exist: " + dir);
        }

        List<Path> files = new ArrayList<>();
        try (Stream<Path> walk = Files.walk(dir)) {
            walk.filter(Files::isRegularFile)
                .filter(p -> p.getFileName().toString().toLowerCase().endsWith(".png"))
                .sorted()
                .forEach(files::add);
        }
        return files;
    }

    private static FingerprintTemplate loadTemplate(Path imagePath, FingerprintImageOptions options) throws IOException {
        return new FingerprintTemplate(
            new FingerprintImage(
                Files.readAllBytes(imagePath),
                options
            )
        );
    }

    private static double scorePair(Path probePath, Path candidatePath, FingerprintImageOptions options) throws IOException {
        FingerprintTemplate probe = loadTemplate(probePath, options);
        FingerprintTemplate candidate = loadTemplate(candidatePath, options);
        return new FingerprintMatcher(probe).match(candidate);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            System.out.println("Usage:");
            System.out.println("mvn exec:java -Dexec.mainClass=LivDetBatchScorer -Dexec.args=\"<trainDir> <trainGlob> <testLiveDir> <testFakeDir> <outputCsv>\"");
            return;
        }

        Path trainDir = Paths.get(args[0]);
        String trainGlob = args[1];
        Path testLiveDir = Paths.get(args[2]);
        Path testFakeDir = Paths.get(args[3]);
        Path outputCsv = Paths.get(args[4]);

        FingerprintImageOptions options = new FingerprintImageOptions().dpi(500);

        List<Path> probes = collectPngs(trainDir, trainGlob);
        List<Path> genuineTargets = collectAllPngs(testLiveDir);
        List<Path> impostorTargets = collectAllPngs(testFakeDir);

        if (probes.isEmpty()) {
            throw new RuntimeException("No probe PNG files found in " + trainDir + " with glob " + trainGlob);
        }
        if (genuineTargets.isEmpty()) {
            throw new RuntimeException("No genuine target PNG files found in " + testLiveDir);
        }
        if (impostorTargets.isEmpty()) {
            throw new RuntimeException("No impostor target PNG files found in " + testFakeDir);
        }

        Files.createDirectories(outputCsv.getParent());

        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outputCsv))) {
            out.println("kind,probe,target,score");

            System.out.println("[i] Probes: " + probes.size());
            System.out.println("[i] Genuine targets: " + genuineTargets.size());
            System.out.println("[i] Impostor targets: " + impostorTargets.size());
            System.out.println("[i] Genuine pairs: " + (probes.size() * genuineTargets.size()));
            System.out.println("[i] Impostor pairs: " + (probes.size() * impostorTargets.size()));

            for (Path probe : probes) {
                for (Path target : genuineTargets) {
                    double score = scorePair(probe, target, options);
                    out.printf(
                        Locale.US,
                        "genuine,%s,%s,%.10f%n",
                        probe.getFileName().toString(),
                        target.getFileName().toString(),
                        score
                    );
                }
            }

            for (Path probe : probes) {
                for (Path target : impostorTargets) {
                    double score = scorePair(probe, target, options);
                    out.printf(
                        Locale.US,
                        "impostor,%s,%s,%.10f%n",
                        probe.getFileName().toString(),
                        target.getFileName().toString(),
                        score
                    );
                }
            }
        }

        System.out.println("[i] Wrote scores to: " + outputCsv);
    }
}