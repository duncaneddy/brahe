package benchmarks;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.DateComponents;
import org.orekit.time.DateTimeComponents;
import org.orekit.time.TimeComponents;
import org.orekit.time.TimeScalesFactory;

/**
 * OreKit time system conversion benchmarks.
 */
public class TimeBenchmark {

    /**
     * Compute Julian Date from OreKit AbsoluteDate components.
     * OreKit doesn't have a direct JD method, so we compute it from date components.
     */
    private static double toJulianDate(AbsoluteDate date, org.orekit.time.TimeScale scale) {
        DateTimeComponents dtc = date.getComponents(scale);
        DateComponents dc = dtc.getDate();
        TimeComponents tc = dtc.getTime();

        int y = dc.getYear();
        int m = dc.getMonth();
        int d = dc.getDay();

        // Julian Day Number using standard algorithm
        int a = (14 - m) / 12;
        int y2 = y + 4800 - a;
        int m2 = m + 12 * a - 3;
        int jdn = d + (153 * m2 + 2) / 5 + 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 - 32045;

        double fracDay = (tc.getHour() - 12) / 24.0
                + tc.getMinute() / 1440.0
                + tc.getSecond() / 86400.0;

        return jdn + fracDay;
    }

    public static JsonObject epochCreation(JsonObject params, int iterations) {
        JsonArray datetimes = params.getAsJsonArray("datetimes");

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (int i = 0; i < datetimes.size(); i++) {
                JsonObject dt = datetimes.get(i).getAsJsonObject();
                int year = dt.get("year").getAsInt();
                int month = dt.get("month").getAsInt();
                int day = dt.get("day").getAsInt();
                int hour = dt.get("hour").getAsInt();
                int minute = dt.get("minute").getAsInt();
                double second = dt.get("second").getAsDouble();
                double nanosecond = dt.get("nanosecond").getAsDouble();

                double totalSeconds = second + nanosecond / 1e9;

                AbsoluteDate date = new AbsoluteDate(year, month, day, hour, minute, totalSeconds,
                        TimeScalesFactory.getUTC());

                double jd = toJulianDate(date, TimeScalesFactory.getUTC());

                if (iter == 0) {
                    iterResults.add(jd);
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

    public static JsonObject utcToTai(JsonObject params, int iterations) {
        JsonArray datetimes = params.getAsJsonArray("datetimes");

        // Pre-build dates
        AbsoluteDate[] dates = buildDates(datetimes);

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (AbsoluteDate date : dates) {
                double jd = toJulianDate(date, TimeScalesFactory.getTAI());
                if (iter == 0) {
                    iterResults.add(jd);
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

    public static JsonObject utcToTt(JsonObject params, int iterations) {
        JsonArray datetimes = params.getAsJsonArray("datetimes");
        AbsoluteDate[] dates = buildDates(datetimes);

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (AbsoluteDate date : dates) {
                double jd = toJulianDate(date, TimeScalesFactory.getTT());
                if (iter == 0) {
                    iterResults.add(jd);
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

    public static JsonObject utcToGps(JsonObject params, int iterations) {
        JsonArray datetimes = params.getAsJsonArray("datetimes");
        AbsoluteDate[] dates = buildDates(datetimes);

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (AbsoluteDate date : dates) {
                double jd = toJulianDate(date, TimeScalesFactory.getGPS());
                if (iter == 0) {
                    iterResults.add(jd);
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

    public static JsonObject utcToUt1(JsonObject params, int iterations) {
        JsonArray datetimes = params.getAsJsonArray("datetimes");
        AbsoluteDate[] dates = buildDates(datetimes);

        JsonArray timesArray = new JsonArray();
        JsonArray resultsArray = new JsonArray();

        for (int iter = 0; iter < iterations; iter++) {
            long start = System.nanoTime();
            JsonArray iterResults = new JsonArray();

            for (AbsoluteDate date : dates) {
                double jd = toJulianDate(date, TimeScalesFactory.getUT1(
                        org.orekit.utils.IERSConventions.IERS_2010, true));
                if (iter == 0) {
                    iterResults.add(jd);
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

    private static AbsoluteDate[] buildDates(JsonArray datetimes) {
        AbsoluteDate[] dates = new AbsoluteDate[datetimes.size()];
        for (int i = 0; i < datetimes.size(); i++) {
            JsonObject dt = datetimes.get(i).getAsJsonObject();
            int year = dt.get("year").getAsInt();
            int month = dt.get("month").getAsInt();
            int day = dt.get("day").getAsInt();
            int hour = dt.get("hour").getAsInt();
            int minute = dt.get("minute").getAsInt();
            double second = dt.get("second").getAsDouble();
            double nanosecond = dt.get("nanosecond").getAsDouble();
            double totalSeconds = second + nanosecond / 1e9;

            dates[i] = new AbsoluteDate(year, month, day, hour, minute, totalSeconds,
                    TimeScalesFactory.getUTC());
        }
        return dates;
    }
}
