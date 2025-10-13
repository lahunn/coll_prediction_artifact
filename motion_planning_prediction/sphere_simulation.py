import sys
import numpy as np
from tqdm import tqdm
import simulation_utils as su

def main():
    if len(sys.argv) != 5:
        print("Usage: python sphere_simulation.py <data_folder> <threshold> <sample_rate> <qnoncoll_len>")
        sys.exit(1)

    data_folder = sys.argv[1]
    threshold = float(sys.argv[2])
    sample_rate = float(sys.argv[3])
    qnoncoll_len = int(sys.argv[4])

    # --- Simulation Settings ---
    # Sphere data is 3D, so we can define a generic binning strategy.
    # This can be adjusted by the user as needed.
    binnumber = 16
    intervalsize = 2 / binnumber
    bins = np.zeros(binnumber)
    start = -1
    for i in range(binnumber):
        bins[i] = start
        start += intervalsize

    fall_prediction = 0
    fall_oracle = 0

    # Assuming a benchmark range, e.g., 2000-2200, similar to other scripts
    for benchid in tqdm(range(2000, 2200)):
        qarr_sphere, _, yarr_sphere = su.load_sphere_data(benchid, data_folder)

        if qarr_sphere is None:
            continue

        # The sphere data is a flat list of spheres, not grouped by path.
        # We can treat the whole file as one large "path" of spheres.

        # Oracle: Count total spheres vs. colliding spheres
        total_spheres = len(yarr_sphere)
        colliding_spheres = total_spheres - np.sum(yarr_sphere)

        # Oracle cost: 1 if there's any collision, otherwise all checks needed.
        all_oracle = 1 if colliding_spheres > 0 else total_spheres

        # The csp_rearrange function is designed for paths of poses.
        # For a flat list of spheres, we can just use the list directly.
        linklist = [item for item in qarr_sphere]
        linklist_coll = [item[0] for item in yarr_sphere]

        # Run the simulation
        all_prediction, _, _ = su.simulate_parallel_collision_detection(
            linklist,
            linklist_coll,
            {}, # Start with an empty collision dictionary for each benchmark
            threshold,
            sample_rate,
            bins,
            qnoncoll_len=qnoncoll_len
        )

        fall_prediction += all_prediction
        fall_oracle += all_oracle

        print(f"Benchmark {benchid}: Prediction Queries={all_prediction}, Oracle Queries={all_oracle}")

    print(f"\n--- Final Results ---")
    print(f"Total Prediction Queries: {fall_prediction}")
    print(f"Total Oracle Queries: {fall_oracle}")

if __name__ == "__main__":
    main()
