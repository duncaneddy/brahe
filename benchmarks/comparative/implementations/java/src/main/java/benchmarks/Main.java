package benchmarks;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.stream.Collectors;

/**
 * CLI dispatcher for OreKit comparative benchmarks.
 * Reads JSON from stdin, runs the benchmark, outputs JSON to stdout.
 */
public class Main {
    private static final Gson gson = new Gson();

    public static void main(String[] args) throws Exception {
        // Initialize OreKit data
        initializeOrekitData();

        // Read JSON from stdin
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String input = reader.lines().collect(Collectors.joining());
        JsonObject inputJson = JsonParser.parseString(input).getAsJsonObject();

        String task = inputJson.get("task").getAsString();
        int iterations = inputJson.get("iterations").getAsInt();
        JsonObject params = inputJson.getAsJsonObject("params");

        JsonObject output;
        switch (task) {
            case "coordinates.geodetic_to_ecef":
                output = CoordinatesBenchmark.geodeticToEcef(params, iterations);
                break;
            case "coordinates.ecef_to_geodetic":
                output = CoordinatesBenchmark.ecefToGeodetic(params, iterations);
                break;
            case "orbits.keplerian_to_cartesian":
                output = OrbitsBenchmark.keplerianToCartesian(params, iterations);
                break;
            case "orbits.cartesian_to_keplerian":
                output = OrbitsBenchmark.cartesianToKeplerian(params, iterations);
                break;
            default:
                System.err.println("Unknown task: " + task);
                System.exit(1);
                return;
        }

        output.addProperty("task", task);
        output.addProperty("iterations", iterations);

        JsonObject metadata = new JsonObject();
        metadata.addProperty("library", "orekit");
        metadata.addProperty("version", "12.2");
        metadata.addProperty("language", "java");
        output.add("metadata", metadata);

        System.out.println(gson.toJson(output));
    }

    private static void initializeOrekitData() {
        String orekitDataPath = System.getenv("OREKIT_DATA");
        if (orekitDataPath == null) {
            orekitDataPath = System.getProperty("user.home") + "/.orekit/orekit-data";
        }

        File orekitData = new File(orekitDataPath);
        if (!orekitData.exists()) {
            System.err.println("OreKit data not found at: " + orekitDataPath);
            System.err.println("Set OREKIT_DATA environment variable or download data to ~/.orekit/orekit-data");
            System.exit(1);
        }

        DataContext.getDefault().getDataProvidersManager()
                .addProvider(new DirectoryCrawler(orekitData));
    }
}
