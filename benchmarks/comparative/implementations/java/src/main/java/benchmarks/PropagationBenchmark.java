package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.ode.nonstiff.ClassicalRungeKuttaIntegrator;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.forces.ForceModel;
import org.orekit.forces.drag.DragForce;
import org.orekit.forces.drag.IsotropicDrag;
import org.orekit.data.DataContext;
import org.orekit.data.DataProvidersManager;
import org.orekit.data.DirectoryCrawler;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.NewtonianAttraction;
import org.orekit.forces.gravity.ThirdBodyAttraction;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.ICGEMFormatReader;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient;
import org.orekit.forces.radiation.SolarRadiationPressure;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.models.earth.atmosphere.Atmosphere;
import org.orekit.models.earth.atmosphere.NRLMSISE00;
import org.orekit.models.earth.atmosphere.data.CssiSpaceWeatherData;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.OrbitType;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.propagation.analytical.tle.TLE;
import org.orekit.propagation.analytical.tle.TLEPropagator;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.DateComponents;
import org.orekit.time.TimeComponents;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;

/**
 * OreKit propagation benchmarks.
 */
public class PropagationBenchmark {

    private static final double MU = Constants.EIGEN5C_EARTH_MU;

    /** Cached spherical-harmonic provider — keyed by (degree, order). When
     *  {@code BRAHE_GRAVITY_FILE} is set to an ICGEM (.gfc) path, the file
     *  is registered with Orekit's data context so {@code
     *  getNormalizedProvider} reads brahe's bundled EGM2008 coefficients
     *  instead of Orekit's default EIGEN-6S. Caching avoids re-registering
     *  the reader (and re-parsing the 7 MB file) on every iteration. */
    private static NormalizedSphericalHarmonicsProvider cachedGravityProvider;
    private static int cachedGravityDegree = -1;
    private static int cachedGravityOrder = -1;

    /** Resolve a normalized spherical-harmonic provider, honoring the
     *  {@code BRAHE_GRAVITY_FILE} env var if set. The file is expected to
     *  be in ICGEM format (.gfc); brahe ships {@code EGM2008_120.gfc} so
     *  the benchmark adapter can point Orekit at the same coefficients
     *  brahe integrates against. */
    private static NormalizedSphericalHarmonicsProvider gravityProvider(int degree, int order) {
        if (cachedGravityProvider != null
                && cachedGravityDegree == degree
                && cachedGravityOrder == order) {
            return cachedGravityProvider;
        }
        String braheGravityFile = System.getenv("BRAHE_GRAVITY_FILE");
        if (braheGravityFile != null && !braheGravityFile.isEmpty()) {
            java.io.File file = new java.io.File(braheGravityFile);
            if (file.exists()) {
                // Register a directory crawler over the gravity-file's parent
                // so Orekit's DataContext can find the ICGEM file by name.
                DataProvidersManager mgr =
                        DataContext.getDefault().getDataProvidersManager();
                mgr.addProvider(new DirectoryCrawler(file.getParentFile()));
                // Clear Orekit's default readers (which would otherwise match
                // ``eigen-6s.gfc`` first) and install only the brahe-bundled
                // ICGEM reader. ``ICGEMFormatReader(name, false)`` takes the
                // exact filename to match and "missing coefficients allowed".
                GravityFieldFactory.clearPotentialCoefficientsReaders();
                GravityFieldFactory.addPotentialCoefficientsReader(
                        new ICGEMFormatReader(file.getName(), false));
            }
        }
        cachedGravityProvider = GravityFieldFactory.getNormalizedProvider(degree, order);
        cachedGravityDegree = degree;
        cachedGravityOrder = order;
        return cachedGravityProvider;
    }
    private static final Frame EME2000 = FramesFactory.getEME2000();
    private static final Frame TEME = FramesFactory.getTEME();
    // brahe integrates in GCRF, so high-fidelity force-model comparisons use GCRF
    // for the inertial frame and ITRF (IERS 2010) for the Earth-fixed force terms.
    private static final Frame GCRF = FramesFactory.getGCRF();
    private static final Frame ITRF =
            FramesFactory.getITRF(IERSConventions.IERS_2010, true);

    /**
     * Convert JD (UTC) to OreKit AbsoluteDate.
     */
    private static AbsoluteDate jdToDate(double jd) {
        double mjd = jd - 2400000.5;
        int mjdDay = (int) Math.floor(mjd);
        double secondsInDay = (mjd - mjdDay) * 86400.0;
        return new AbsoluteDate(
                new DateComponents(DateComponents.MODIFIED_JULIAN_EPOCH, mjdDay),
                new TimeComponents(secondsInDay),
                TimeScalesFactory.getUTC());
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
        double stepSize = params.get("step_size").getAsDouble();
        int nSteps = params.get("n_steps").getAsInt();

        // Two parameter shapes (see python/propagation.py docstring):
        //   - perf:     {jd, elements, step_size, n_steps}
        //   - accuracy: {cases: [{jd, elements}], step_size, n_steps}
        // The perf path emits every trajectory step's state. The accuracy
        // path emits one final state per case so the harness can compute
        // distribution stats across independent initial conditions.
        JsonArray cases = params.has("cases") ? params.getAsJsonArray("cases") : null;

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            if (cases == null) {
                AbsoluteDate epoch = jdToDate(params.get("jd").getAsDouble());
                JsonArray oe = params.getAsJsonArray("elements");
                KeplerianOrbit orbit = new KeplerianOrbit(
                        oe.get(0).getAsDouble(),
                        oe.get(1).getAsDouble(),
                        Math.toRadians(oe.get(2).getAsDouble()),
                        Math.toRadians(oe.get(4).getAsDouble()),
                        Math.toRadians(oe.get(3).getAsDouble()),
                        Math.toRadians(oe.get(5).getAsDouble()),
                        PositionAngleType.MEAN, EME2000, epoch, MU);
                KeplerianPropagator prop = new KeplerianPropagator(orbit);
                for (int step = 1; step <= nSteps; step++) {
                    AbsoluteDate target = epoch.shiftedBy(step * stepSize);
                    SpacecraftState state = prop.propagate(target);
                    if (iter == 0) addState(iterResults, state.getPVCoordinates(EME2000));
                }
            } else {
                for (int c = 0; c < cases.size(); c++) {
                    JsonObject cas = cases.get(c).getAsJsonObject();
                    AbsoluteDate epoch = jdToDate(cas.get("jd").getAsDouble());
                    JsonArray oe = cas.getAsJsonArray("elements");
                    KeplerianOrbit orbit = new KeplerianOrbit(
                            oe.get(0).getAsDouble(),
                            oe.get(1).getAsDouble(),
                            Math.toRadians(oe.get(2).getAsDouble()),
                            Math.toRadians(oe.get(4).getAsDouble()),
                            Math.toRadians(oe.get(3).getAsDouble()),
                            Math.toRadians(oe.get(5).getAsDouble()),
                            PositionAngleType.MEAN, EME2000, epoch, MU);
                    KeplerianPropagator prop = new KeplerianPropagator(orbit);
                    AbsoluteDate target = epoch.shiftedBy(nSteps * stepSize);
                    SpacecraftState state = prop.propagate(target);
                    if (iter == 0) addState(iterResults, state.getPVCoordinates(EME2000));
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
        double stepSize = params.get("step_size").getAsDouble();
        int nSteps = params.get("n_steps").getAsInt();

        // perf:     {jd, elements, step_size, n_steps}      -> trajectory
        // accuracy: {cases:[{jd,elements}], step_size, n_steps} -> final state per case
        JsonArray cases = params.has("cases") ? params.getAsJsonArray("cases") : null;

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            if (cases == null) {
                AbsoluteDate epoch = jdToDate(params.get("jd").getAsDouble());
                JsonArray oe = params.getAsJsonArray("elements");
                KeplerianOrbit initialOrbit = new KeplerianOrbit(
                        oe.get(0).getAsDouble(),
                        oe.get(1).getAsDouble(),
                        Math.toRadians(oe.get(2).getAsDouble()),
                        Math.toRadians(oe.get(4).getAsDouble()),
                        Math.toRadians(oe.get(3).getAsDouble()),
                        Math.toRadians(oe.get(5).getAsDouble()),
                        PositionAngleType.MEAN, EME2000, epoch, MU);
                ClassicalRungeKuttaIntegrator integrator =
                        new ClassicalRungeKuttaIntegrator(stepSize);
                NumericalPropagator prop = new NumericalPropagator(integrator);
                prop.setOrbitType(OrbitType.CARTESIAN);
                prop.addForceModel(new NewtonianAttraction(MU));
                prop.setInitialState(new SpacecraftState(initialOrbit));
                for (int step = 1; step <= nSteps; step++) {
                    AbsoluteDate target = epoch.shiftedBy(step * stepSize);
                    SpacecraftState state = prop.propagate(target);
                    if (iter == 0) addState(iterResults, state.getPVCoordinates(EME2000));
                }
            } else {
                for (int c = 0; c < cases.size(); c++) {
                    JsonObject cas = cases.get(c).getAsJsonObject();
                    AbsoluteDate epoch = jdToDate(cas.get("jd").getAsDouble());
                    JsonArray oe = cas.getAsJsonArray("elements");
                    KeplerianOrbit initialOrbit = new KeplerianOrbit(
                            oe.get(0).getAsDouble(),
                            oe.get(1).getAsDouble(),
                            Math.toRadians(oe.get(2).getAsDouble()),
                            Math.toRadians(oe.get(4).getAsDouble()),
                            Math.toRadians(oe.get(3).getAsDouble()),
                            Math.toRadians(oe.get(5).getAsDouble()),
                            PositionAngleType.MEAN, EME2000, epoch, MU);
                    ClassicalRungeKuttaIntegrator integrator =
                            new ClassicalRungeKuttaIntegrator(stepSize);
                    NumericalPropagator prop = new NumericalPropagator(integrator);
                    prop.setOrbitType(OrbitType.CARTESIAN);
                    prop.addForceModel(new NewtonianAttraction(MU));
                    prop.setInitialState(new SpacecraftState(initialOrbit));
                    AbsoluteDate target = epoch.shiftedBy(nSteps * stepSize);
                    SpacecraftState state = prop.propagate(target);
                    if (iter == 0) addState(iterResults, state.getPVCoordinates(EME2000));
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

    // =========================================================================
    // RK4 force-model propagation benchmarks
    //
    // These three methods share a single helper (numericalRk4Run) and only
    // differ in which optional force terms (third body, drag, SRP) are enabled
    // based on the input params. They mirror the brahe NumericalOrbitPropagator
    // configurations so trajectories can be compared point-by-point.
    // =========================================================================

    public static JsonObject numericalRk4Grav5x5(JsonObject params, int iterations) {
        return numericalRk4Run(params, iterations);
    }

    public static JsonObject numericalRk4Grav20x20SunMoon(JsonObject params, int iterations) {
        return numericalRk4Run(params, iterations);
    }

    public static JsonObject numericalRk4Grav80x80Full(JsonObject params, int iterations) {
        return numericalRk4Run(params, iterations);
    }

    private static JsonObject numericalRk4Run(JsonObject params, int iterations) {
        double stepSize = params.get("step_size").getAsDouble();
        int nSteps = params.get("n_steps").getAsInt();

        int gravityDegree = params.get("gravity_degree").getAsInt();
        int gravityOrder = params.get("gravity_order").getAsInt();
        boolean thirdBodySun = params.has("third_body_sun") && params.get("third_body_sun").getAsBoolean();
        boolean thirdBodyMoon = params.has("third_body_moon") && params.get("third_body_moon").getAsBoolean();
        boolean drag = params.has("drag") && params.get("drag").getAsBoolean();
        boolean srp = params.has("srp") && params.get("srp").getAsBoolean();

        // brahe parameter vector layout: [mass, drag_area, Cd, srp_area, Cr]
        JsonArray paramsArray = params.getAsJsonArray("params");
        double mass = paramsArray.get(0).getAsDouble();
        double dragArea = paramsArray.get(1).getAsDouble();
        double cd = paramsArray.get(2).getAsDouble();
        double srpArea = paramsArray.get(3).getAsDouble();
        double cr = paramsArray.get(4).getAsDouble();

        // Use the gravity provider's mu (matches whatever ICGEM file is
        // selected — EGM2008 when BRAHE_GRAVITY_FILE points at brahe's
        // bundled file, EIGEN-6S otherwise) so the Keplerian-to-Cartesian
        // conversion is consistent with the central term used during
        // integration.
        final NormalizedSphericalHarmonicsProvider gravityProvider =
                gravityProvider(gravityDegree, gravityOrder);
        final double gm = gravityProvider.getMu();

        // Earth shape for drag/SRP.
        final OneAxisEllipsoid earthShape =
                new OneAxisEllipsoid(
                        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                        Constants.WGS84_EARTH_FLATTENING,
                        ITRF);

        final Atmosphere atmosphere;
        if (drag) {
            CssiSpaceWeatherData sw = new CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt");
            atmosphere = new NRLMSISE00(sw, CelestialBodyFactory.getSun(), earthShape);
        } else {
            atmosphere = null;
        }

        // perf:     {jd, elements_deg, ...}             -> trajectory output
        // accuracy: {cases:[{jd,elements}], ...}        -> final state per case
        JsonArray cases = params.has("cases") ? params.getAsJsonArray("cases") : null;

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            if (cases == null) {
                double jd = params.get("jd").getAsDouble();
                JsonArray oe = params.getAsJsonArray("elements_deg");
                AbsoluteDate epoch = jdToDate(jd);
                KeplerianOrbit initialOrbit = new KeplerianOrbit(
                        oe.get(0).getAsDouble(),
                        oe.get(1).getAsDouble(),
                        Math.toRadians(oe.get(2).getAsDouble()),
                        Math.toRadians(oe.get(4).getAsDouble()),
                        Math.toRadians(oe.get(3).getAsDouble()),
                        Math.toRadians(oe.get(5).getAsDouble()),
                        PositionAngleType.MEAN, GCRF, epoch, gm);
                NumericalPropagator prop = buildRk4Propagator(
                        stepSize, gravityProvider, atmosphere, earthShape,
                        thirdBodySun, thirdBodyMoon, drag, srp,
                        dragArea, cd, srpArea, cr);
                prop.setInitialState(new SpacecraftState(initialOrbit, mass));
                for (int step = 1; step <= nSteps; step++) {
                    AbsoluteDate target = epoch.shiftedBy(step * stepSize);
                    SpacecraftState state = prop.propagate(target);
                    if (iter == 0) addState(iterResults, state.getPVCoordinates(GCRF));
                }
            } else {
                for (int c = 0; c < cases.size(); c++) {
                    JsonObject cas = cases.get(c).getAsJsonObject();
                    AbsoluteDate epoch = jdToDate(cas.get("jd").getAsDouble());
                    JsonArray oe = cas.getAsJsonArray("elements");
                    KeplerianOrbit initialOrbit = new KeplerianOrbit(
                            oe.get(0).getAsDouble(),
                            oe.get(1).getAsDouble(),
                            Math.toRadians(oe.get(2).getAsDouble()),
                            Math.toRadians(oe.get(4).getAsDouble()),
                            Math.toRadians(oe.get(3).getAsDouble()),
                            Math.toRadians(oe.get(5).getAsDouble()),
                            PositionAngleType.MEAN, GCRF, epoch, gm);
                    NumericalPropagator prop = buildRk4Propagator(
                            stepSize, gravityProvider, atmosphere, earthShape,
                            thirdBodySun, thirdBodyMoon, drag, srp,
                            dragArea, cd, srpArea, cr);
                    prop.setInitialState(new SpacecraftState(initialOrbit, mass));
                    AbsoluteDate target = epoch.shiftedBy(nSteps * stepSize);
                    SpacecraftState state = prop.propagate(target);
                    if (iter == 0) addState(iterResults, state.getPVCoordinates(GCRF));
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

    /** Build a fresh RK4 propagator with the same force-model stack used by
     * both the perf path (one IC) and the accuracy path (per-case rebuild).
     */
    private static NumericalPropagator buildRk4Propagator(
            double stepSize,
            NormalizedSphericalHarmonicsProvider gravityProvider,
            Atmosphere atmosphere,
            OneAxisEllipsoid earthShape,
            boolean thirdBodySun,
            boolean thirdBodyMoon,
            boolean drag,
            boolean srp,
            double dragArea,
            double cd,
            double srpArea,
            double cr) {
        ClassicalRungeKuttaIntegrator integrator =
                new ClassicalRungeKuttaIntegrator(stepSize);
        NumericalPropagator prop = new NumericalPropagator(integrator);
        prop.setOrbitType(OrbitType.CARTESIAN);

        // Spherical-harmonic gravity in the ITRF body-fixed frame.
        prop.addForceModel(new HolmesFeatherstoneAttractionModel(ITRF, gravityProvider));

        if (thirdBodySun) {
            prop.addForceModel(new ThirdBodyAttraction(CelestialBodyFactory.getSun()));
        }
        if (thirdBodyMoon) {
            prop.addForceModel(new ThirdBodyAttraction(CelestialBodyFactory.getMoon()));
        }
        if (drag) {
            prop.addForceModel(new DragForce(atmosphere, new IsotropicDrag(dragArea, cd)));
        }
        if (srp) {
            prop.addForceModel(new SolarRadiationPressure(
                    CelestialBodyFactory.getSun(), earthShape,
                    new IsotropicRadiationSingleCoefficient(srpArea, cr)));
        }
        return prop;
    }
}
