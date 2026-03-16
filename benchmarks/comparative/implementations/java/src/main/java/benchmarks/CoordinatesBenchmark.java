package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.FramesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;

/**
 * OreKit coordinate conversion benchmarks.
 */
public class CoordinatesBenchmark {

    private static OneAxisEllipsoid getEarth() {
        return new OneAxisEllipsoid(
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                Constants.WGS84_EARTH_FLATTENING,
                FramesFactory.getITRF(IERSConventions.IERS_2010, true));
    }

    public static JsonObject geodeticToEcef(JsonObject params, int iterations) {
        OneAxisEllipsoid earth = getEarth();
        JsonArray points = params.getAsJsonArray("points");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < points.size(); i++) {
                JsonArray pt = points.get(i).getAsJsonArray();
                double lon = Math.toRadians(pt.get(0).getAsDouble());
                double lat = Math.toRadians(pt.get(1).getAsDouble());
                double alt = pt.get(2).getAsDouble();

                GeodeticPoint geodeticPt = new GeodeticPoint(lat, lon, alt);
                Vector3D ecef = earth.transform(geodeticPt);

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(ecef.getX());
                    result.add(ecef.getY());
                    result.add(ecef.getZ());
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

    public static JsonObject ecefToGeodetic(JsonObject params, int iterations) {
        OneAxisEllipsoid earth = getEarth();
        JsonArray points = params.getAsJsonArray("points");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < points.size(); i++) {
                JsonArray pt = points.get(i).getAsJsonArray();
                double x = pt.get(0).getAsDouble();
                double y = pt.get(1).getAsDouble();
                double z = pt.get(2).getAsDouble();

                Vector3D position = new Vector3D(x, y, z);
                GeodeticPoint geodeticPt = earth.transform(position, earth.getBodyFrame(), null);

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(Math.toDegrees(geodeticPt.getLongitude()));
                    result.add(Math.toDegrees(geodeticPt.getLatitude()));
                    result.add(geodeticPt.getAltitude());
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
