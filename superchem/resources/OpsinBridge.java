// SUPERChem MIT-licensed bridge. OPSIN is an independently MIT-licensed dependency.
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import uk.ac.cam.ch.wwmm.opsin.NameToStructure;
import uk.ac.cam.ch.wwmm.opsin.NameToStructureConfig;
import uk.ac.cam.ch.wwmm.opsin.OpsinResult;

public class OpsinBridge {
    private static String encode(String value) {
        return Base64.getEncoder().encodeToString((value == null ? "" : value).getBytes(StandardCharsets.UTF_8));
    }
    public static void main(String[] args) throws Exception {
        NameToStructure parser = NameToStructure.getInstance();
        NameToStructureConfig config = new NameToStructureConfig();
        config.setAllowRadicals(false);
        config.setWarnRatherThanFailOnUninterpretableStereochemistry(false);
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        while ((line = input.readLine()) != null) {
            try {
                String name = new String(Base64.getDecoder().decode(line), StandardCharsets.UTF_8);
                OpsinResult result = parser.parseChemicalName(name, config);
                System.out.println(result.getStatus().toString() + "\t" + encode(result.getSmiles()) + "\t" + encode(result.getMessage()));
            } catch (Exception e) {
                System.out.println("FAILURE\t\t" + encode(e.getClass().getSimpleName() + ": " + e.getMessage()));
            }
        }
    }
}
