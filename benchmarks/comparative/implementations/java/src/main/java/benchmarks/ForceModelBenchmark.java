package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.NewtonianAttraction;
import org.orekit.forces.gravity.ThirdBodyAttraction;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.propagation.SpacecraftState;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.DateComponents;
import org.orekit.time.TimeComponents;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;
import org.orekit.utils.TimeStampedPVCoordinates;

/**
 * OreKit function-level force-model acceleration benchmarks.
 *
 * <p>Each task evaluates a single acceleration term at a fixed state and epoch,
 * so the comparison with brahe isolates the force-model implementation from
 * the propagator and integrator.
 */
public class ForceModelBenchmark {

    private static final double MU = Constants.EIGEN5C_EARTH_MU;
    private static final Frame GCRF = FramesFactory.getGCRF();
    private static final Frame ITRF =
            FramesFactory.getITRF(IERSConventions.IERS_2010, true);

    private static AbsoluteDate jdToDate(double jd) {
        double mjd = jd - 2400000.5;
        int mjdDay = (int) Math.floor(mjd);
        double secondsInDay = (mjd - mjdDay) * 86400.0;
        return new AbsoluteDate(
                new DateComponents(DateComponents.MODIFIED_JULIAN_EPOCH, mjdDay),
                new TimeComponents(secondsInDay),
                TimeScalesFactory.getUTC());
    }

    private static void addAccel(JsonArray results, Vector3D a) {
        JsonArray row = new JsonArray();
        row.add(a.getX());
        row.add(a.getY());
        row.add(a.getZ());
        results.add(row);
    }

    private static SpacecraftState buildState(AbsoluteDate epoch, JsonArray state) {
        Vector3D pos = new Vector3D(
                state.get(0).getAsDouble(),
                state.get(1).getAsDouble(),
                state.get(2).getAsDouble());
        Vector3D vel = new Vector3D(
                state.get(3).getAsDouble(),
                state.get(4).getAsDouble(),
                state.get(5).getAsDouble());
        TimeStampedPVCoordinates pv = new TimeStampedPVCoordinates(epoch, pos, vel, Vector3D.ZERO);
        CartesianOrbit orbit = new CartesianOrbit(pv, GCRF, MU);
        return new SpacecraftState(orbit);
    }

    public static JsonObject accelPointMassGravity(JsonObject params, int iterations) {
        double jd = params.get("jd").getAsDouble();
        JsonArray stateArr = params.getAsJsonArray("state_eci");
        int nSamples = params.get("n_samples").getAsInt();

        AbsoluteDate epoch = jdToDate(jd);
        SpacecraftState scState = buildState(epoch, stateArr);

        // Use Orekit's native NewtonianAttraction force model so the timed work
        // is the library's own point-mass evaluation, matching the pattern used
        // by the spherical-harmonics and third-body tasks below.
        NewtonianAttraction gravity = new NewtonianAttraction(MU);
        gravity.init(scState, scState.getDate());
        double[] modelParams = gravity.getParameters(scState.getDate());

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            Vector3D a = Vector3D.ZERO;
            for (int j = 0; j < nSamples; j++) {
                a = gravity.acceleration(scState, modelParams);
            }
            double elapsed = (System.nanoTime() - start) / 1e9;
            timesArray.add(elapsed);
            if (iter == 0) {
                addAccel(resultsArray, a);
            }
        }

        JsonObject output = new JsonObject();
        output.add("times_seconds", timesArray);
        output.add("results", resultsArray);
        return output;
    }

    private static JsonObject accelSphericalHarmonicsRun(JsonObject params, int iterations) {
        double jd = params.get("jd").getAsDouble();
        JsonArray stateArr = params.getAsJsonArray("state_eci");
        int nSamples = params.get("n_samples").getAsInt();
        int degree = params.get("degree").getAsInt();
        int order = params.get("order").getAsInt();

        AbsoluteDate epoch = jdToDate(jd);
        SpacecraftState scState = buildState(epoch, stateArr);

        NormalizedSphericalHarmonicsProvider provider =
                GravityFieldFactory.getNormalizedProvider(degree, order);
        HolmesFeatherstoneAttractionModel gravity =
                new HolmesFeatherstoneAttractionModel(ITRF, provider);
        gravity.init(scState, scState.getDate());
        double[] modelParams = gravity.getParameters(scState.getDate());
        // Orekit's HolmesFeatherstoneAttractionModel returns the non-central
        // perturbations only. brahe's accel_gravity_spherical_harmonics returns
        // the full gravity including the central term. To make the comparison
        // apples-to-apples we add the central GM/r^2 term to Orekit's output.
        final double providerMu = provider.getMu();
        final Vector3D position = scState.getPosition();

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            Vector3D a = Vector3D.ZERO;
            for (int j = 0; j < nSamples; j++) {
                Vector3D aPert = gravity.acceleration(scState, modelParams);
                double r3 = position.getNorm() * position.getNorm() * position.getNorm();
                Vector3D aCentral = position.scalarMultiply(-providerMu / r3);
                a = aPert.add(aCentral);
            }
            double elapsed = (System.nanoTime() - start) / 1e9;
            timesArray.add(elapsed);
            if (iter == 0) {
                addAccel(resultsArray, a);
            }
        }

        JsonObject output = new JsonObject();
        output.add("times_seconds", timesArray);
        output.add("results", resultsArray);
        return output;
    }

    public static JsonObject accelSphericalHarmonics20(JsonObject params, int iterations) {
        return accelSphericalHarmonicsRun(params, iterations);
    }

    public static JsonObject accelSphericalHarmonics80(JsonObject params, int iterations) {
        return accelSphericalHarmonicsRun(params, iterations);
    }

    private static JsonObject accelThirdBodyRun(
            JsonObject params, int iterations, CelestialBody body) {
        double jd = params.get("jd").getAsDouble();
        JsonArray stateArr = params.getAsJsonArray("state_eci");
        int nSamples = params.get("n_samples").getAsInt();

        AbsoluteDate epoch = jdToDate(jd);
        SpacecraftState scState = buildState(epoch, stateArr);

        ThirdBodyAttraction force = new ThirdBodyAttraction(body);
        force.init(scState, scState.getDate());
        double[] modelParams = force.getParameters(scState.getDate());

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            Vector3D a = Vector3D.ZERO;
            for (int j = 0; j < nSamples; j++) {
                a = force.acceleration(scState, modelParams);
            }
            double elapsed = (System.nanoTime() - start) / 1e9;
            timesArray.add(elapsed);
            if (iter == 0) {
                addAccel(resultsArray, a);
            }
        }

        JsonObject output = new JsonObject();
        output.add("times_seconds", timesArray);
        output.add("results", resultsArray);
        return output;
    }

    public static JsonObject accelThirdBodySun(JsonObject params, int iterations) {
        return accelThirdBodyRun(params, iterations, CelestialBodyFactory.getSun());
    }

    public static JsonObject accelThirdBodyMoon(JsonObject params, int iterations) {
        return accelThirdBodyRun(params, iterations, CelestialBodyFactory.getMoon());
    }
}
