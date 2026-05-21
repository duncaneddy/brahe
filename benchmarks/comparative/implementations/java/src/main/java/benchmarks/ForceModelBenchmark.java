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
import org.orekit.utils.TimeStampedPVCoordinates;

/**
 * OreKit function-level force-model acceleration benchmarks.
 *
 * <p>Each task supports two input shapes:
 * <ul>
 *   <li>Perf (single IC): one fixed state, inner loop of {@code n_samples}
 *       repetitions for amortized timing — returns one acceleration vector.</li>
 *   <li>Accuracy (multi-IC): a {@code cases} array of {jd, state_eci} ICs,
 *       evaluated once each — returns one acceleration per case so the
 *       accuracy harness can compare per-sample and build a distribution.</li>
 * </ul>
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

    /** Functional shape of "evaluate this force model at this spacecraft state". */
    @FunctionalInterface
    private interface AccelEval {
        Vector3D evaluate(SpacecraftState state);
    }

    /** Factory: per-state construction of the force-model evaluator. */
    @FunctionalInterface
    private interface EvalFactory {
        AccelEval forState(SpacecraftState state);
    }

    /**
     * Single dispatch point for the perf vs. accuracy paths. Each task hands in
     * an {@link EvalFactory} that constructs the per-state evaluator (because
     * Orekit force models cache state-dependent parameters in their
     * {@code init()} call, the evaluator must be rebuilt per case on the
     * accuracy path).
     */
    private static JsonObject runForceModel(JsonObject params, int iterations, EvalFactory factory) {
        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        if (params.has("cases")) {
            // Accuracy path: one evaluation per case.
            JsonArray cases = params.getAsJsonArray("cases");
            for (int iter = 0; iter < iterations; iter++) {
                long start = System.nanoTime();
                JsonArray iterResults = new JsonArray();
                for (int c = 0; c < cases.size(); c++) {
                    JsonObject caseObj = cases.get(c).getAsJsonObject();
                    AbsoluteDate epoch = jdToDate(caseObj.get("jd").getAsDouble());
                    SpacecraftState scState =
                            buildState(epoch, caseObj.getAsJsonArray("state_eci"));
                    AccelEval eval = factory.forState(scState);
                    Vector3D a = eval.evaluate(scState);
                    addAccel(iterResults, a);
                }
                double elapsed = (System.nanoTime() - start) / 1e9;
                timesArray.add(elapsed);
                if (iter == 0) {
                    resultsArray = iterResults;
                }
            }
        } else {
            // Perf path: single state, inner loop of n_samples for timing.
            double jd = params.get("jd").getAsDouble();
            JsonArray stateArr = params.getAsJsonArray("state_eci");
            int nSamples = params.get("n_samples").getAsInt();
            AbsoluteDate epoch = jdToDate(jd);
            SpacecraftState scState = buildState(epoch, stateArr);
            AccelEval eval = factory.forState(scState);

            for (int iter = 0; iter < iterations; iter++) {
                long start = System.nanoTime();
                Vector3D a = Vector3D.ZERO;
                for (int j = 0; j < nSamples; j++) {
                    a = eval.evaluate(scState);
                }
                double elapsed = (System.nanoTime() - start) / 1e9;
                timesArray.add(elapsed);
                if (iter == 0) {
                    addAccel(resultsArray, a);
                }
            }
        }

        JsonObject output = new JsonObject();
        output.add("times_seconds", timesArray);
        output.add("results", resultsArray);
        return output;
    }

    public static JsonObject accelPointMassGravity(JsonObject params, int iterations) {
        // Use Orekit's native NewtonianAttraction so the timed work is the
        // library's own point-mass evaluation, matching the pattern used by
        // the spherical-harmonics and third-body tasks below.
        return runForceModel(params, iterations, scState -> {
            NewtonianAttraction gravity = new NewtonianAttraction(MU);
            gravity.init(scState, scState.getDate());
            double[] modelParams = gravity.getParameters(scState.getDate());
            return s -> gravity.acceleration(s, modelParams);
        });
    }

    private static JsonObject accelSphericalHarmonicsRun(JsonObject params, int iterations) {
        int degree = params.get("degree").getAsInt();
        int order = params.get("order").getAsInt();

        NormalizedSphericalHarmonicsProvider provider =
                GravityFieldFactory.getNormalizedProvider(degree, order);
        // Orekit's HolmesFeatherstoneAttractionModel returns the non-central
        // perturbations only; brahe returns the full gravity including the central
        // term. We add the central GM/r^2 back to Orekit's output to compare apples-to-apples.
        final double providerMu = provider.getMu();

        return runForceModel(params, iterations, scState -> {
            HolmesFeatherstoneAttractionModel gravity =
                    new HolmesFeatherstoneAttractionModel(ITRF, provider);
            gravity.init(scState, scState.getDate());
            double[] modelParams = gravity.getParameters(scState.getDate());
            return s -> {
                Vector3D aPert = gravity.acceleration(s, modelParams);
                Vector3D position = s.getPosition();
                double r3 = position.getNorm() * position.getNorm() * position.getNorm();
                Vector3D aCentral = position.scalarMultiply(-providerMu / r3);
                return aPert.add(aCentral);
            };
        });
    }

    public static JsonObject accelSphericalHarmonics20(JsonObject params, int iterations) {
        return accelSphericalHarmonicsRun(params, iterations);
    }

    public static JsonObject accelSphericalHarmonics80(JsonObject params, int iterations) {
        return accelSphericalHarmonicsRun(params, iterations);
    }

    private static JsonObject accelThirdBodyRun(
            JsonObject params, int iterations, CelestialBody body) {
        return runForceModel(params, iterations, scState -> {
            ThirdBodyAttraction force = new ThirdBodyAttraction(body);
            force.init(scState, scState.getDate());
            double[] modelParams = force.getParameters(scState.getDate());
            return s -> force.acceleration(s, modelParams);
        });
    }

    public static JsonObject accelThirdBodySun(JsonObject params, int iterations) {
        return accelThirdBodyRun(params, iterations, CelestialBodyFactory.getSun());
    }

    public static JsonObject accelThirdBodyMoon(JsonObject params, int iterations) {
        return accelThirdBodyRun(params, iterations, CelestialBodyFactory.getMoon());
    }
}
