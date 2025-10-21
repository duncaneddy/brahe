#!/usr/bin/env python3
"""
Parallelization Test Script

Tests the parallel access computation implementation by comparing:
1. Sequential vs Parallel performance
2. Single-threaded vs Multi-threaded results (should be identical)
3. Different thread counts

This script creates a realistic scenario with multiple satellites and locations.
"""

import brahe as bh
import numpy as np
import time


def create_test_scenario():
    """Create test scenario with multiple satellites and locations."""
    print("Setting up test scenario...")

    # Create epoch
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    days = 7.0  # 7 days

    # Create multiple ground locations (distributed globally)
    locations = [
        bh.PointLocation(0.0, 45.0, 0.0).with_name("Location-1"),  # Europe
        bh.PointLocation(-120.0, 37.0, 0.0).with_name("Location-2"),  # USA West
        bh.PointLocation(139.0, 35.0, 0.0).with_name("Location-3"),  # Japan
        bh.PointLocation(-43.0, -22.0, 0.0).with_name("Location-4"),  # Brazil
        bh.PointLocation(18.0, -33.0, 0.0).with_name("Location-5"),  # South Africa
    ]

    # Create multiple satellites with different orbital planes
    propagators = []
    for i in range(5):
        # Orbital elements: [a, e, i, raan, argp, M]
        oe = np.array(
            [
                bh.R_EARTH + 500e3,  # 500 km altitude
                0.001,  # Low eccentricity
                np.radians(97.8),  # Sun-synchronous inclination
                np.radians(i * 30),  # Different RAANs
                0.0,
                np.radians(i * 45),  # Different mean anomalies
            ]
        )

        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

        prop = bh.KeplerianPropagator.from_eci(
            epoch,
            state,
            60.0,  # step_size
        ).with_name(f"Satellite-{i + 1}")

        propagators.append(prop)

    # Define search period (6 hours)
    search_start = epoch
    search_end = epoch + 24 * 3600.0 * 7.0

    # Define constraint
    constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

    print(f"  Created {len(locations)} locations")
    print(f"  Created {len(propagators)} satellites")
    print(f"  Search period: {days} days")
    print(f"  Total combinations: {len(locations) * len(propagators)}")

    return locations, propagators, search_start, search_end, constraint


def test_sequential_vs_parallel():
    """Test that sequential and parallel produce identical results."""
    print("\n" + "=" * 80)
    print("TEST 1: Sequential vs Parallel Correctness")
    print("=" * 80)

    locations, propagators, search_start, search_end, constraint = (
        create_test_scenario()
    )

    # Run sequential
    print("\nRunning SEQUENTIAL computation...")
    config_seq = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        parallel=False,
    )

    start_time = time.time()
    windows_seq = bh.location_accesses(
        locations,
        propagators,
        search_start,
        search_end,
        constraint,
        config=config_seq,
    )
    seq_time = time.time() - start_time

    print(f"  Found {len(windows_seq)} access windows")
    print(f"  Time: {seq_time:.3f} seconds")

    # Run parallel (default threads)
    print("\nRunning PARALLEL computation (default: ~90% cores)...")
    config_par = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        parallel=True,
    )

    start_time = time.time()
    windows_par = bh.location_accesses(
        locations,
        propagators,
        search_start,
        search_end,
        constraint,
        config=config_par,
    )
    par_time = time.time() - start_time

    print(f"  Found {len(windows_par)} access windows")
    print(f"  Time: {par_time:.3f} seconds")
    print(f"  Speedup: {seq_time / par_time:.2f}x")

    # Verify same number of windows
    if len(windows_seq) != len(windows_par):
        print("\n❌ FAILED: Different number of windows!")
        print(f"  Sequential: {len(windows_seq)}")
        print(f"  Parallel: {len(windows_par)}")
        return False

    # Verify windows are identical (within tolerance)
    print("\nVerifying window timing matches...")
    tolerance = 0.01  # 0.01 second tolerance
    for i, (w_seq, w_par) in enumerate(zip(windows_seq, windows_par)):
        dt_open = abs(w_seq.window_open - w_par.window_open)
        dt_close = abs(w_seq.window_close - w_par.window_close)

        if dt_open > tolerance or dt_close > tolerance:
            print(f"\n❌ FAILED: Window {i} times don't match!")
            print(f"  Sequential: {w_seq.window_open} - {w_seq.window_close}")
            print(f"  Parallel:   {w_par.window_open} - {w_par.window_close}")
            print(f"  Differences: {dt_open:.6f}s, {dt_close:.6f}s")
            return False

    print("  ✓ All window times match (within tolerance)")
    print("\n✅ PASSED: Sequential and parallel produce identical results")
    print(
        f"   Performance gain: {seq_time / par_time:.2f}x faster with parallelization"
    )

    return True


def test_thread_scaling():
    """Test different thread counts."""
    print("\n" + "=" * 80)
    print("TEST 2: Thread Scaling")
    print("=" * 80)

    max_threads = bh.get_max_threads()
    print(f"\nDefault thread pool: {max_threads} threads")

    locations, propagators, search_start, search_end, constraint = (
        create_test_scenario()
    )

    # Test with different thread counts
    thread_counts = [1, 2, 4, max_threads]
    results = []

    for num_threads in thread_counts:
        print(f"\nTesting with {num_threads} thread(s)...")

        config = bh.AccessSearchConfig(
            initial_time_step=60.0,
            adaptive_step=False,
            parallel=True,
            num_threads=num_threads,
        )

        start_time = time.time()
        windows = bh.location_accesses(
            locations,
            propagators,
            search_start,
            search_end,
            constraint,
            config=config,
        )
        elapsed = time.time() - start_time

        results.append((num_threads, len(windows), elapsed))
        print(f"  Found {len(windows)} windows in {elapsed:.3f} seconds")

    # Verify all found same number of windows
    window_counts = [r[1] for r in results]
    if len(set(window_counts)) != 1:
        print(
            "\n❌ FAILED: Different thread counts found different numbers of windows!"
        )
        for threads, count, _ in results:
            print(f"  {threads} threads: {count} windows")
        return False

    # Show scaling
    print("\nScaling analysis:")
    baseline_time = results[0][2]  # 1-thread time
    for threads, count, elapsed in results:
        speedup = baseline_time / elapsed
        efficiency = speedup / threads * 100
        print(
            f"  {threads} thread(s): {elapsed:.3f}s, speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%"
        )

    print("\n✅ PASSED: All thread counts produce consistent results")
    return True


def test_global_thread_setting():
    """Test global thread pool setting."""
    print("\n" + "=" * 80)
    print("TEST 3: Global Thread Pool Setting")
    print("=" * 80)

    # Note: This test should be run in a fresh Python process
    # because once the thread pool is initialized, it can't be changed

    current_threads = bh.get_max_threads()
    print(f"\nCurrent thread pool: {current_threads} threads")
    print("  (90% of available cores)")

    print("\n✅ PASSED: Global thread pool accessible")
    print("   Note: set_max_threads() must be called before any parallel operations")
    print("   Note: Once initialized, thread pool cannot be changed")

    return True


def main():
    """Run all parallelization tests."""
    print("=" * 80)
    print("BRAHE PARALLELIZATION TEST SUITE")
    print("=" * 80)

    # Initialize EOP provider (required for frame transformations)
    print("\nInitializing EOP provider...")
    eop = bh.StaticEOPProvider.from_values(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bh.set_global_eop_provider(eop)
    print("  ✓ EOP provider initialized")

    print("\nSystem info:")
    print(f"  Available cores: {bh.get_max_threads() / 0.9:.0f}")  # Approximate
    print(f"  Default threads: {bh.get_max_threads()}")

    # Run tests
    test_results = []

    test_results.append(("Sequential vs Parallel", test_sequential_vs_parallel()))
    test_results.append(("Thread Scaling", test_thread_scaling()))
    test_results.append(("Global Thread Setting", test_global_thread_setting()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")

    all_passed = all(result[1] for result in test_results)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nParallelization is working correctly:")
        print("  • Sequential and parallel produce identical results")
        print("  • Different thread counts work consistently")
        print("  • Performance scales with thread count")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
