package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.Transform;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;

/**
 * OreKit frame transformation benchmarks.
 */
public class FramesBenchmark {

    private static final Frame GCRF = FramesFactory.getGCRF();
    private static final Frame ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, true);

    /**
     * Convert JD to OreKit AbsoluteDate.
     */
    private static AbsoluteDate jdToDate(double jd) {
        // J2000.0 epoch is JD 2451545.0
        double offsetDays = jd - 2451545.0;
        double offsetSeconds = offsetDays * 86400.0;
        return AbsoluteDate.J2000_EPOCH.shiftedBy(offsetSeconds);
    }

    public static JsonObject stateEciToEcef(JsonObject params, int iterations) {
        JsonArray cases = params.getAsJsonArray("cases");

        // Pre-build dates and PV coordinates
        int n = cases.size();
        AbsoluteDate[] dates = new AbsoluteDate[n];
        PVCoordinates[] pvs = new PVCoordinates[n];

        for (int i = 0; i < n; i++) {
            JsonObject c = cases.get(i).getAsJsonObject();
            double jd = c.get("jd").getAsDouble();
            JsonArray state = c.getAsJsonArray("state");

            dates[i] = jdToDate(jd);
            Vector3D pos = new Vector3D(
                    state.get(0).getAsDouble(),
                    state.get(1).getAsDouble(),
                    state.get(2).getAsDouble());
            Vector3D vel = new Vector3D(
                    state.get(3).getAsDouble(),
                    state.get(4).getAsDouble(),
                    state.get(5).getAsDouble());
            pvs[i] = new PVCoordinates(pos, vel);
        }

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < n; i++) {
                Transform transform = GCRF.getTransformTo(ITRF, dates[i]);
                PVCoordinates ecef = transform.transformPVCoordinates(pvs[i]);

                if (iter == 0) {
                    Vector3D p = ecef.getPosition();
                    Vector3D v = ecef.getVelocity();
                    JsonArray result = new JsonArray();
                    result.add(p.getX());
                    result.add(p.getY());
                    result.add(p.getZ());
                    result.add(v.getX());
                    result.add(v.getY());
                    result.add(v.getZ());
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

    public static JsonObject stateEcefToEci(JsonObject params, int iterations) {
        JsonArray cases = params.getAsJsonArray("cases");

        int n = cases.size();
        AbsoluteDate[] dates = new AbsoluteDate[n];
        PVCoordinates[] pvs = new PVCoordinates[n];

        for (int i = 0; i < n; i++) {
            JsonObject c = cases.get(i).getAsJsonObject();
            double jd = c.get("jd").getAsDouble();
            JsonArray state = c.getAsJsonArray("state");

            dates[i] = jdToDate(jd);
            Vector3D pos = new Vector3D(
                    state.get(0).getAsDouble(),
                    state.get(1).getAsDouble(),
                    state.get(2).getAsDouble());
            Vector3D vel = new Vector3D(
                    state.get(3).getAsDouble(),
                    state.get(4).getAsDouble(),
                    state.get(5).getAsDouble());
            pvs[i] = new PVCoordinates(pos, vel);
        }

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < n; i++) {
                Transform transform = ITRF.getTransformTo(GCRF, dates[i]);
                PVCoordinates eci = transform.transformPVCoordinates(pvs[i]);

                if (iter == 0) {
                    Vector3D p = eci.getPosition();
                    Vector3D v = eci.getVelocity();
                    JsonArray result = new JsonArray();
                    result.add(p.getX());
                    result.add(p.getY());
                    result.add(p.getZ());
                    result.add(v.getX());
                    result.add(v.getY());
                    result.add(v.getZ());
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
