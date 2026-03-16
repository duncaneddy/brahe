package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.TopocentricFrame;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.tle.TLE;
import org.orekit.propagation.analytical.tle.TLEPropagator;
import org.orekit.propagation.events.EventDetector;
import org.orekit.propagation.events.ElevationDetector;
import org.orekit.propagation.events.handlers.EventHandler;
import org.hipparchus.ode.events.Action;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;

import java.util.ArrayList;
import java.util.List;

/**
 * OreKit access computation benchmarks.
 */
public class AccessBenchmark {

    private static final Frame ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, true);

    private static final OneAxisEllipsoid EARTH = new OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_FLATTENING,
            ITRF);

    private static AbsoluteDate jdToDate(double jd) {
        double offsetDays = jd - 2451545.0;
        double offsetSeconds = offsetDays * 86400.0;
        return AbsoluteDate.J2000_EPOCH.shiftedBy(offsetSeconds);
    }

    public static JsonObject sgp4Access(JsonObject params, int iterations) {
        String line1 = params.get("line1").getAsString();
        String line2 = params.get("line2").getAsString();
        double minElDeg = params.get("min_elevation_deg").getAsDouble();
        double duration = params.get("search_duration_seconds").getAsDouble();
        JsonArray locationsJson = params.getAsJsonArray("locations");

        TLE tle = new TLE(line1, line2);
        AbsoluteDate searchStart = tle.getDate();
        AbsoluteDate searchEnd = searchStart.shiftedBy(duration);

        // Build topocentric frames for each location
        List<TopocentricFrame> topoFrames = new ArrayList<>();
        for (int i = 0; i < locationsJson.size(); i++) {
            JsonObject loc = locationsJson.get(i).getAsJsonObject();
            double lon = Math.toRadians(loc.get("lon").getAsDouble());
            double lat = Math.toRadians(loc.get("lat").getAsDouble());
            double alt = loc.get("alt").getAsDouble();

            GeodeticPoint point = new GeodeticPoint(lat, lon, alt);
            TopocentricFrame topo = new TopocentricFrame(EARTH, point, "loc_" + i);
            topoFrames.add(topo);
        }

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int locIdx = 0; locIdx < topoFrames.size(); locIdx++) {
                TopocentricFrame topo = topoFrames.get(locIdx);
                JsonArray locWindows = new JsonArray();

                // Create fresh propagator per location (event handler state is per-location)
                TLEPropagator prop = TLEPropagator.selectExtrapolator(tle);

                // Set up elevation detector
                List<double[]> windows = new ArrayList<>();
                final double[] riseTime = {Double.NaN};

                ElevationDetector detector = new ElevationDetector(30.0, 0.001, topo)
                        .withConstantElevation(Math.toRadians(minElDeg))
                        .withHandler(new EventHandler() {
                            @Override
                            public Action eventOccurred(SpacecraftState s, EventDetector d, boolean increasing) {
                                if (increasing) {
                                    // Rise event
                                    riseTime[0] = s.getDate().durationFrom(AbsoluteDate.J2000_EPOCH);
                                } else {
                                    // Set event
                                    if (!Double.isNaN(riseTime[0])) {
                                        double setTimeSec = s.getDate().durationFrom(AbsoluteDate.J2000_EPOCH);
                                        windows.add(new double[]{riseTime[0], setTimeSec});
                                        riseTime[0] = Double.NaN;
                                    }
                                }
                                return Action.CONTINUE;
                            }
                        });

                prop.addEventDetector(detector);
                prop.propagate(searchStart, searchEnd);

                // Convert to JD windows
                for (double[] window : windows) {
                    double startJd = window[0] / 86400.0 + 2451545.0;
                    double endJd = window[1] / 86400.0 + 2451545.0;
                    JsonObject w = new JsonObject();
                    w.addProperty("start_jd", startJd);
                    w.addProperty("end_jd", endJd);
                    locWindows.add(w);
                }

                if (iter == 0) {
                    iterResults.add(locWindows);
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
