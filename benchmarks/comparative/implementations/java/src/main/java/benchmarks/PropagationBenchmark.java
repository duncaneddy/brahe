package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.orekit.forces.gravity.NewtonianAttraction;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.OrbitType;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.propagation.analytical.tle.TLE;
import org.orekit.propagation.analytical.tle.TLEPropagator;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.PVCoordinates;

/**
 * OreKit propagation benchmarks.
 */
public class PropagationBenchmark {

    private static final double MU = Constants.EIGEN5C_EARTH_MU;
    private static final Frame EME2000 = FramesFactory.getEME2000();
    private static final Frame TEME = FramesFactory.getTEME();

    private static AbsoluteDate jdToDate(double jd) {
        double offsetDays = jd - 2451545.0;
        double offsetSeconds = offsetDays * 86400.0;
        return AbsoluteDate.J2000_EPOCH.shiftedBy(offsetSeconds);
    }

    private static void addState(JsonArray results, PVCoordinates pv) {
        Vector3D pos = pv.getPosition();
        Vector3D vel = pv.getVelocity();
        JsonArray result = new JsonArray();
        result.add(pos.getX());
        result.add(pos.getY());
        result.add(pos.getZ());
        result.add(vel.getX());
        result.add(vel.getY());
        result.add(vel.getZ());
        results.add(result);
    }

    public static JsonObject keplerianSingle(JsonObject params, int iterations) {
        JsonArray cases = params.getAsJsonArray("cases");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < cases.size(); i++) {
                JsonObject c = cases.get(i).getAsJsonObject();
                double jd = c.get("jd").getAsDouble();
                JsonArray oe = c.getAsJsonArray("elements");
                double dt = c.get("dt").getAsDouble();

                AbsoluteDate epoch = jdToDate(jd);
                AbsoluteDate target = epoch.shiftedBy(dt);

                double a = oe.get(0).getAsDouble();
                double e = oe.get(1).getAsDouble();
                double inc = Math.toRadians(oe.get(2).getAsDouble());
                double raan = Math.toRadians(oe.get(3).getAsDouble());
                double argp = Math.toRadians(oe.get(4).getAsDouble());
                double M = Math.toRadians(oe.get(5).getAsDouble());

                KeplerianOrbit orbit = new KeplerianOrbit(
                        a, e, inc, argp, raan, M,
                        PositionAngleType.MEAN, EME2000, epoch, MU);
                KeplerianPropagator prop = new KeplerianPropagator(orbit);
                SpacecraftState finalState = prop.propagate(target);
                PVCoordinates pv = finalState.getPVCoordinates(EME2000);

                if (iter == 0) {
                    addState(iterResults, pv);
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

    public static JsonObject keplerianTrajectory(JsonObject params, int iterations) {
        double jd = params.get("jd").getAsDouble();
        JsonArray oe = params.getAsJsonArray("elements");
        double stepSize = params.get("step_size").getAsDouble();
        int nSteps = params.get("n_steps").getAsInt();

        AbsoluteDate epoch = jdToDate(jd);
        double a = oe.get(0).getAsDouble();
        double e = oe.get(1).getAsDouble();
        double inc = Math.toRadians(oe.get(2).getAsDouble());
        double raan = Math.toRadians(oe.get(3).getAsDouble());
        double argp = Math.toRadians(oe.get(4).getAsDouble());
        double M = Math.toRadians(oe.get(5).getAsDouble());

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            KeplerianOrbit orbit = new KeplerianOrbit(
                    a, e, inc, argp, raan, M,
                    PositionAngleType.MEAN, EME2000, epoch, MU);
            KeplerianPropagator prop = new KeplerianPropagator(orbit);

            for (int step = 1; step <= nSteps; step++) {
                AbsoluteDate target = epoch.shiftedBy(step * stepSize);
                SpacecraftState state = prop.propagate(target);
                PVCoordinates pv = state.getPVCoordinates(EME2000);

                if (iter == 0) {
                    addState(iterResults, pv);
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

    public static JsonObject sgp4Single(JsonObject params, int iterations) {
        String line1 = params.get("line1").getAsString();
        String line2 = params.get("line2").getAsString();
        JsonArray offsets = params.getAsJsonArray("time_offsets_seconds");

        TLE tle = new TLE(line1, line2);
        AbsoluteDate baseEpoch = tle.getDate();

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            TLEPropagator prop = TLEPropagator.selectExtrapolator(tle);

            for (int i = 0; i < offsets.size(); i++) {
                double dt = offsets.get(i).getAsDouble();
                AbsoluteDate target = baseEpoch.shiftedBy(dt);
                SpacecraftState state = prop.propagate(target);
                PVCoordinates pv = state.getPVCoordinates(TEME);

                if (iter == 0) {
                    addState(iterResults, pv);
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

    public static JsonObject sgp4Trajectory(JsonObject params, int iterations) {
        String line1 = params.get("line1").getAsString();
        String line2 = params.get("line2").getAsString();
        double stepSize = params.get("step_size").getAsDouble();
        int nSteps = params.get("n_steps").getAsInt();

        TLE tle = new TLE(line1, line2);
        AbsoluteDate baseEpoch = tle.getDate();

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            TLEPropagator prop = TLEPropagator.selectExtrapolator(tle);

            for (int step = 1; step <= nSteps; step++) {
                AbsoluteDate target = baseEpoch.shiftedBy(step * stepSize);
                SpacecraftState state = prop.propagate(target);
                PVCoordinates pv = state.getPVCoordinates(TEME);

                if (iter == 0) {
                    addState(iterResults, pv);
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

    public static JsonObject numericalTwobody(JsonObject params, int iterations) {
        double jd = params.get("jd").getAsDouble();
        JsonArray oe = params.getAsJsonArray("elements");
        double stepSize = params.get("step_size").getAsDouble();
        int nSteps = params.get("n_steps").getAsInt();

        AbsoluteDate epoch = jdToDate(jd);
        double a = oe.get(0).getAsDouble();
        double e = oe.get(1).getAsDouble();
        double inc = Math.toRadians(oe.get(2).getAsDouble());
        double raan = Math.toRadians(oe.get(3).getAsDouble());
        double argp = Math.toRadians(oe.get(4).getAsDouble());
        double M = Math.toRadians(oe.get(5).getAsDouble());

        // Create initial orbit from Keplerian elements
        KeplerianOrbit initialOrbit = new KeplerianOrbit(
                a, e, inc, argp, raan, M,
                PositionAngleType.MEAN, EME2000, epoch, MU);

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            // Create numerical propagator with Dormand-Prince 8(5,3) integrator
            double minStep = 0.001;
            double maxStep = 300.0;
            double positionError = 1.0; // 1 meter position error tolerance
            DormandPrince853Integrator integrator = new DormandPrince853Integrator(
                    minStep, maxStep, positionError, positionError);
            NumericalPropagator prop = new NumericalPropagator(integrator);
            prop.setOrbitType(OrbitType.CARTESIAN);

            // Add only Newtonian (two-body) gravity
            prop.addForceModel(new NewtonianAttraction(MU));
            prop.setInitialState(new SpacecraftState(initialOrbit));

            for (int step = 1; step <= nSteps; step++) {
                AbsoluteDate target = epoch.shiftedBy(step * stepSize);
                SpacecraftState state = prop.propagate(target);
                PVCoordinates pv = state.getPVCoordinates(EME2000);

                if (iter == 0) {
                    addState(iterResults, pv);
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
