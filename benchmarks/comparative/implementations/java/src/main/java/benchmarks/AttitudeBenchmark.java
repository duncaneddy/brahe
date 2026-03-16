package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.RotationConvention;
import org.hipparchus.geometry.euclidean.threed.RotationOrder;

/**
 * OreKit/Hipparchus attitude conversion benchmarks.
 */
public class AttitudeBenchmark {

    public static JsonObject quaternionToRotationMatrix(JsonObject params, int iterations) {
        JsonArray quaternions = params.getAsJsonArray("quaternions");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < quaternions.size(); i++) {
                JsonArray q = quaternions.get(i).getAsJsonArray();
                double w = q.get(0).getAsDouble();
                double x = q.get(1).getAsDouble();
                double y = q.get(2).getAsDouble();
                double z = q.get(3).getAsDouble();

                // Hipparchus Rotation: (q0=w, q1=x, q2=y, q3=z, needsNormalization)
                Rotation rot = new Rotation(w, x, y, z, false);
                double[][] mat = rot.getMatrix();

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    // Row-major flattening
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 3; c++) {
                            result.add(mat[r][c]);
                        }
                    }
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

    public static JsonObject rotationMatrixToQuaternion(JsonObject params, int iterations) {
        JsonArray matrices = params.getAsJsonArray("matrices");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < matrices.size(); i++) {
                JsonArray matJson = matrices.get(i).getAsJsonArray();
                double[][] mat = new double[3][3];
                for (int r = 0; r < 3; r++) {
                    JsonArray row = matJson.get(r).getAsJsonArray();
                    for (int c = 0; c < 3; c++) {
                        mat[r][c] = row.get(c).getAsDouble();
                    }
                }

                Rotation rot = new Rotation(mat, 1e-10);
                // Extract quaternion components [w, x, y, z]
                double w = rot.getQ0();
                double x = rot.getQ1();
                double y = rot.getQ2();
                double z = rot.getQ3();

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(w);
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

    public static JsonObject quaternionToEulerAngle(JsonObject params, int iterations) {
        JsonArray quaternions = params.getAsJsonArray("quaternions");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < quaternions.size(); i++) {
                JsonArray q = quaternions.get(i).getAsJsonArray();
                double w = q.get(0).getAsDouble();
                double x = q.get(1).getAsDouble();
                double y = q.get(2).getAsDouble();
                double z = q.get(3).getAsDouble();

                Rotation rot = new Rotation(w, x, y, z, false);
                double[][] mat = rot.getMatrix();

                // Extract ZYX Euler angles using Brahe's column-based convention:
                // R = Rz(phi) * Ry(theta) * Rx(psi)
                double phi = Math.atan2(-mat[1][0], mat[0][0]);
                double theta = Math.asin(mat[2][0]);
                double psi = Math.atan2(-mat[2][1], mat[2][2]);

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(phi);   // Z rotation
                    result.add(theta); // Y rotation
                    result.add(psi);   // X rotation
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

    public static JsonObject eulerAngleToQuaternion(JsonObject params, int iterations) {
        JsonArray angles = params.getAsJsonArray("angles");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < angles.size(); i++) {
                JsonArray a = angles.get(i).getAsJsonArray();
                double phi = a.get(0).getAsDouble();
                double theta = a.get(1).getAsDouble();
                double psi = a.get(2).getAsDouble();

                // Build rotation matrix using Brahe's convention: R = Rz(phi) * Ry(theta) * Rx(psi)
                double cp = Math.cos(phi), sp = Math.sin(phi);
                double ct = Math.cos(theta), st = Math.sin(theta);
                double cs = Math.cos(psi), ss = Math.sin(psi);

                double[][] mat = {
                    { cp*ct,  cp*st*ss - sp*cs,  cp*st*cs + sp*ss },
                    { sp*ct,  sp*st*ss + cp*cs,  sp*st*cs - cp*ss },
                    { -st,    ct*ss,             ct*cs             }
                };

                Rotation rot = new Rotation(mat, 1e-10);

                if (iter == 0) {
                    JsonArray result = new JsonArray();
                    result.add(rot.getQ0()); // w
                    result.add(rot.getQ1()); // x
                    result.add(rot.getQ2()); // y
                    result.add(rot.getQ3()); // z
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
