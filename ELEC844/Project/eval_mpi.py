import csv
import sys
import os
import statistics


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <filename.csv>")
        return

    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found.")
        return

    # Containers for our data
    # We will dynamically populate these based on headers
    metrics_data = {}
    imbalance_values = []

    try:
        with open(filename, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # Identify which columns are metrics vs processor work
            process_cols = [h for h in headers if h.startswith(
                'process ') and h.endswith(' work')]
            metric_cols = [
                h for h in headers if 'process' not in h.lower() and h.strip()]

            # Initialize lists for metrics
            for col in metric_cols:
                metrics_data[col] = []

            for row in reader:
                # 1. Process Standard Metrics (Length, Time, etc.)
                for col in metric_cols:
                    try:
                        val = float(row[col])
                        metrics_data[col].append(val)
                    except (ValueError, TypeError):
                        continue

                # 2. Process Workload Imbalance (Max - Min of processes)
                try:
                    p_values = [float(row[p])
                                for p in process_cols if row[p].strip()]
                    if p_values:
                        diff = max(p_values) - min(p_values)
                        imbalance_values.append(diff)
                except (ValueError, TypeError):
                    continue

        # --- Output Results ---
        print(f"Analysis for: {filename}")
        print(f"Processors detected: {len(process_cols)}")
        print("-" * 60)
        print(f"{'Metric':<30} | {'Median Value':<15}")
        print("-" * 60)

        # Print medians for standard metrics
        for col in metric_cols:
            data = metrics_data[col]
            if data:
                med = statistics.median(data)
                print(f"{col:<30} | {med:<15.4f}")
            else:
                print(f"{col:<30} | {'No Data':<15}")

        # Print the Workload Imbalance metric
        print("-" * 60)
        if imbalance_values:
            imbalance_median = statistics.median(imbalance_values)
            print(f"{'Workload Imbalance (Median)':<30} | {
                  imbalance_median:<15.4f}")
        else:
            print(f"{'Workload Imbalance (Median)':<30} | {'No Data':<15}")
        print("-" * 60)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
