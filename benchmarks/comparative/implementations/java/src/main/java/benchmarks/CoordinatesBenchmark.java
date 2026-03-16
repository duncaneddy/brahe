package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.TopocentricFrame;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
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

    public static JsonObject geocentricToEcef(JsonObject params, int iterations) {
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

                // Geocentric spherical to Cartesian (altitude convention: r = R_EARTH + alt)
                double r = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + alt;
                double x = r * Math.cos(lat) * Math.cos(lon);
                double y = r * Math.cos(lat) * Math.sin(lon);
                double z = r * Math.sin(lat);

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(x);
                    result.add(y);
                    result.add(z);
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

    public static JsonObject ecefToGeocentric(JsonObject params, int iterations) {
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

                double radius = Math.sqrt(x * x + y * y + z * z);
                double lon = Math.toDegrees(Math.atan2(y, x));
                double lat = Math.toDegrees(Math.asin(z / radius));
                double alt = radius - Constants.WGS84_EARTH_EQUATORIAL_RADIUS;

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(lon);
                    result.add(lat);
                    result.add(alt);
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

    public static JsonObject ecefToAzel(JsonObject params, int iterations) {
        OneAxisEllipsoid earth = getEarth();
        JsonArray pairs = params.getAsJsonArray("pairs");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        // Use a reference date (doesn't affect ECEF-based computation)
        AbsoluteDate refDate = new AbsoluteDate(2024, 1, 1, 0, 0, 0.0,
                TimeScalesFactory.getUTC());

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < pairs.size(); i++) {
                JsonObject pair = pairs.get(i).getAsJsonObject();
                JsonArray stationGeodetic = pair.getAsJsonArray("station_geodetic");
                JsonArray satelliteEcef = pair.getAsJsonArray("satellite_ecef");

                double staLon = Math.toRadians(stationGeodetic.get(0).getAsDouble());
                double staLat = Math.toRadians(stationGeodetic.get(1).getAsDouble());
                double staAlt = stationGeodetic.get(2).getAsDouble();

                GeodeticPoint stationPoint = new GeodeticPoint(staLat, staLon, staAlt);
                TopocentricFrame topoFrame = new TopocentricFrame(earth, stationPoint, "station");

                double satX = satelliteEcef.get(0).getAsDouble();
                double satY = satelliteEcef.get(1).getAsDouble();
                double satZ = satelliteEcef.get(2).getAsDouble();
                Vector3D satPos = new Vector3D(satX, satY, satZ);

                double az = Math.toDegrees(topoFrame.getAzimuth(satPos, earth.getBodyFrame(), refDate));
                double el = Math.toDegrees(topoFrame.getElevation(satPos, earth.getBodyFrame(), refDate));
                double range = topoFrame.getRange(satPos, earth.getBodyFrame(), refDate);

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(az);
                    result.add(el);
                    result.add(range);
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
