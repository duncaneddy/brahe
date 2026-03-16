package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.PVCoordinates;

/**
 * OreKit orbital element conversion benchmarks.
 */
public class OrbitsBenchmark {

    private static final double MU = Constants.EIGEN5C_EARTH_MU;
    private static final AbsoluteDate EPOCH =
            new AbsoluteDate(2024, 1, 1, 0, 0, 0.0, TimeScalesFactory.getUTC());

    public static JsonObject keplerianToCartesian(JsonObject params, int iterations) {
        JsonArray elements = params.getAsJsonArray("elements");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < elements.size(); i++) {
                JsonArray oe = elements.get(i).getAsJsonArray();
                double a = oe.get(0).getAsDouble();
                double e = oe.get(1).getAsDouble();
                double inc = Math.toRadians(oe.get(2).getAsDouble());
                double raan = Math.toRadians(oe.get(3).getAsDouble());
                double argp = Math.toRadians(oe.get(4).getAsDouble());
                double M = Math.toRadians(oe.get(5).getAsDouble());

                KeplerianOrbit orbit = new KeplerianOrbit(
                        a, e, inc, argp, raan, M,
                        PositionAngleType.MEAN,
                        FramesFactory.getEME2000(), EPOCH, MU);

                PVCoordinates pv = orbit.getPVCoordinates();
                Vector3D pos = pv.getPosition();
                Vector3D vel = pv.getVelocity();

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(pos.getX());
                    result.add(pos.getY());
                    result.add(pos.getZ());
                    result.add(vel.getX());
                    result.add(vel.getY());
                    result.add(vel.getZ());
                    iterResults.add(result);
                }
            }

            double elapsed = (System.nanoTime() - start) / 1e9;
            timesArray.add(elapsed);

            if (iter == 0) {
                resultsArray = iterResults;
            }
        }

        JsonObject output = new JsonObject();
        output.add("times_seconds", timesArray);
        output.add("results", resultsArray);
        return output;
    }

    public static JsonObject cartesianToKeplerian(JsonObject params, int iterations) {
        JsonArray states = params.getAsJsonArray("states");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < states.size(); i++) {
                JsonArray state = states.get(i).getAsJsonArray();
                Vector3D pos = new Vector3D(
                        state.get(0).getAsDouble(),
                        state.get(1).getAsDouble(),
                        state.get(2).getAsDouble());
                Vector3D vel = new Vector3D(
                        state.get(3).getAsDouble(),
                        state.get(4).getAsDouble(),
                        state.get(5).getAsDouble());

                PVCoordinates pv = new PVCoordinates(pos, vel);
                CartesianOrbit cartOrbit = new CartesianOrbit(
                        pv, FramesFactory.getEME2000(), EPOCH, MU);
                KeplerianOrbit kepOrbit = new KeplerianOrbit(cartOrbit);

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(kepOrbit.getA());
                    result.add(kepOrbit.getE());
                    result.add(Math.toDegrees(kepOrbit.getI()));
                    result.add(Math.toDegrees(kepOrbit.getRightAscensionOfAscendingNode()));
                    result.add(Math.toDegrees(kepOrbit.getPerigeeArgument()));
                    result.add(Math.toDegrees(kepOrbit.getMeanAnomaly()));
                    iterResults.add(result);
                }
            }

            double elapsed = (System.nanoTime() - start) / 1e9;
            timesArray.add(elapsed);

            if (iter == 0) {
                resultsArray = iterResults;
            }
        }

        JsonObject output = new JsonObject();
        output.add("times_seconds", timesArray);
        output.add("results", resultsArray);
        return output;
    }
}
