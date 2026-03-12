import com.machinezoo.sourceafis.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Main {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Usage: mvn exec:java -Dexec.mainClass=Main -Dexec.args=\"probe.png candidate.png\"");
            return;
        }

        String probePath = args[0];
        String candidatePath = args[1];

        var options = new FingerprintImageOptions().dpi(500);

        var probe = new FingerprintTemplate(
            new FingerprintImage(
                Files.readAllBytes(Paths.get(probePath)),
                options
            )
        );

        var candidate = new FingerprintTemplate(
            new FingerprintImage(
                Files.readAllBytes(Paths.get(candidatePath)),
                options
            )
        );

        double score = new FingerprintMatcher(probe).match(candidate);
        boolean matches = score >= 40;

        System.out.println("Probe file: " + probePath);
        System.out.println("Candidate file: " + candidatePath);
        System.out.println("Similarity score: " + score);
        System.out.println("Match at threshold 40? " + matches);
    }
}