import com.machinezoo.sourceafis.*;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class BatchScorer {
    private static final Pattern NAME_RE = Pattern.compile("^(\\d+)_(\\d+)\\.png$", Pattern.CASE_INSENSITIVE);
    private static final long RANDOM_SEED = 123L;

    private static class TemplateFile {
        Path path;
        int fingerId;
        int sampleId;

        TemplateFile(Path path, int fingerId, int sampleId) {
            this.path = path;
            this.fingerId = fingerId;
            this.sampleId = sampleId;
        }
    }

    private static TemplateFile parseTemplate(Path p) {
        String name = p.getFileName().toString();
        Matcher m = NAME_RE.matcher(name);
        if (!m.matches()) {
            return null;
        }
        int fingerId = Integer.parseInt(m.group(1));
        int sampleId = Integer.parseInt(m.group(2));
        return new TemplateFile(p, fingerId, sampleId);
    }

    private static List<TemplateFile> collectTemplates(Path dataDir) throws IOException {
        List<TemplateFile> files = new ArrayList<>();
        try (Stream<Path> walk = Files.walk(dataDir)) {
            walk.filter(Files::isRegularFile)
                .filter(p -> p.getFileName().toString().toLowerCase().endsWith(".png"))
                .sorted()
                .forEach(p -> {
                    TemplateFile tf = parseTemplate(p);
                    if (tf != null) {
                        files.add(tf);
                    }
                });
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

    private static Set<Integer> parseProbeSamples(String csv) {
        Set<Integer> samples = new HashSet<>();
        for (String s : csv.split(",")) {
            s = s.trim();
            if (!s.isEmpty()) {
                samples.add(Integer.parseInt(s));
            }
        }
        return samples;
    }

    private static class PairEntry {
        TemplateFile a;
        TemplateFile b;

        PairEntry(TemplateFile a, TemplateFile b) {
            this.a = a;
            this.b = b;
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 4 || args.length > 5) {
            System.out.println("Usage: mvn exec:java -Dexec.mainClass=BatchScorer -Dexec.args=\"<dataDir> <userId> <probeSamplesCsv> <outputCsv> [maxImpostorScores]\"");
            return;
        }

        Path dataDir = Paths.get(args[0]);
        int userId = Integer.parseInt(args[1]);
        Set<Integer> probeSamples = parseProbeSamples(args[2]);
        Path outputCsv = Paths.get(args[3]);
        Integer maxImpostorScores = null;
        if (args.length == 5) {
            int v = Integer.parseInt(args[4]);
            if (v >= 0) {
                maxImpostorScores = v;
            }
        }

        FingerprintImageOptions options = new FingerprintImageOptions().dpi(500);

        List<TemplateFile> all = collectTemplates(dataDir);
        if (all.isEmpty()) {
            throw new RuntimeException("No PNG files found under: " + dataDir);
        }

        Map<Integer, List<TemplateFile>> byId = new TreeMap<>();
        for (TemplateFile tf : all) {
            byId.computeIfAbsent(tf.fingerId, k -> new ArrayList<>()).add(tf);
        }

        if (!byId.containsKey(userId)) {
            throw new RuntimeException("USER_ID=" + userId + " not found in dataset.");
        }

        List<TemplateFile> allUser = new ArrayList<>(byId.get(userId));
        allUser.sort(Comparator.comparingInt(t -> t.sampleId));

        List<TemplateFile> probeTemplates = new ArrayList<>();
        List<TemplateFile> genuineTargets = new ArrayList<>();

        for (TemplateFile tf : allUser) {
            if (probeSamples.contains(tf.sampleId)) {
                probeTemplates.add(tf);
            } else {
                genuineTargets.add(tf);
            }
        }

        if (probeTemplates.size() != probeSamples.size()) {
            List<Integer> existing = new ArrayList<>();
            for (TemplateFile tf : allUser) {
                existing.add(tf.sampleId);
            }
            throw new RuntimeException("Missing requested probe samples. Existing samples for user "
                + userId + ": " + existing);
        }

        if (genuineTargets.isEmpty()) {
            throw new RuntimeException("No remaining genuine targets after excluding probe samples.");
        }

        List<TemplateFile> otherTemplates = new ArrayList<>();
        for (Map.Entry<Integer, List<TemplateFile>> e : byId.entrySet()) {
            if (e.getKey() != userId) {
                otherTemplates.addAll(e.getValue());
            }
        }

        List<PairEntry> genuinePairs = new ArrayList<>();
        for (TemplateFile p : probeTemplates) {
            for (TemplateFile g : genuineTargets) {
                genuinePairs.add(new PairEntry(p, g));
            }
        }

        List<PairEntry> impostorPairs = new ArrayList<>();
        for (TemplateFile p : probeTemplates) {
            for (TemplateFile o : otherTemplates) {
                impostorPairs.add(new PairEntry(p, o));
            }
        }

        if (maxImpostorScores != null && impostorPairs.size() > maxImpostorScores) {
            Collections.shuffle(impostorPairs, new Random(RANDOM_SEED));
            impostorPairs = new ArrayList<>(impostorPairs.subList(0, maxImpostorScores));
        }

        Files.createDirectories(outputCsv.getParent());

        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outputCsv))) {
            out.println("kind,probe,target,score");

            System.out.println("[i] Probe templates: " + probeTemplates.size());
            System.out.println("[i] Genuine targets: " + genuineTargets.size());
            System.out.println("[i] Genuine pairs: " + genuinePairs.size());
            System.out.println("[i] Impostor pairs: " + impostorPairs.size());

            for (PairEntry pair : genuinePairs) {
                double score = scorePair(pair.a.path, pair.b.path, options);
                out.printf(
                    Locale.US,
                    "genuine,%s,%s,%.10f%n",
                    pair.a.path.getFileName().toString(),
                    pair.b.path.getFileName().toString(),
                    score
                );
            }

            for (PairEntry pair : impostorPairs) {
                double score = scorePair(pair.a.path, pair.b.path, options);
                out.printf(
                    Locale.US,
                    "impostor,%s,%s,%.10f%n",
                    pair.a.path.getFileName().toString(),
                    pair.b.path.getFileName().toString(),
                    score
                );
            }
        }

        System.out.println("[i] Wrote scores to: " + outputCsv);
    }
}