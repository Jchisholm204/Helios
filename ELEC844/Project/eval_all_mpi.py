import csv
import sys
import os
import statistics
import re


def get_stats(filename):
    """Processes a single CSV and returns a dictionary of medians."""
    metrics_data = {}
    imbalance_values = []

    try:
        with open(filename, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if not headers:
                return None

            process_cols = [h for h in headers if h.startswith(
                'process ') and h.endswith(' work')]
            metric_cols = [
                h for h in headers if 'process' not in h.lower() and h.strip()]

            for col in metric_cols:
                metrics_data[col] = []

            for row in reader:
                # Standard Metrics
                for col in metric_cols:
                    try:
                        metrics_data[col].append(float(row[col]))
                    except:
                        continue
                # Workload Imbalance
                try:
                    p_vals = [float(row[p])
                              for p in process_cols if row[p].strip()]
                    if p_vals:
                        imbalance_values.append(max(p_vals) - min(p_vals))
                except:
                    continue

        # Calculate medians for this specific file
        results = {col: statistics.median(
            data) if data else 0.0 for col, data in metrics_data.items()}
        if imbalance_values:
            results['Workload Imbalance'] = statistics.median(imbalance_values)

        return results
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    # Pattern to match: astar_{dims}_p{procs}.csv
    pattern = re.compile(r"astar_(\d+D)_p(\d+)\.csv")

    # Grouping structure: { "4D": { 16: stats, 32: stats }, "6D": {...} }
    dimension_groups = {}

    for file in os.listdir(target_dir):
        match = pattern.match(file)
        if match:
            dims, procs = match.groups()
            procs = int(procs)

            file_path = os.path.join(target_dir, file)
            stats = get_stats(file_path)

            if stats:
                if dims not in dimension_groups:
                    dimension_groups[dims] = {}
                dimension_groups[dims][procs] = stats

    # Output one table per dimension
    for dims in sorted(dimension_groups.keys()):
        print(f"\n{'='*20} DIMENSION: {dims} {'='*20}")

        procs_list = sorted(dimension_groups[dims].keys())
        # Get all unique metric names found in these files
        all_metrics = []
        for p in procs_list:
            for m in dimension_groups[dims][p].keys():
                if m not in all_metrics:
                    all_metrics.append(m)

        # Header Row: Metric Name | p2 | p4 | p8 ...
        header = f"{'Metric':<30}"
        for p in procs_list:
            header += f" | p{p:<10}"
        print(header)
        print("-" * len(header))

        # Data Rows
        for metric in all_metrics:
            row_str = f"{metric:<30}"
            for p in procs_list:
                val = dimension_groups[dims][p].get(metric, "N/A")
                if isinstance(val, float):
                    row_str += f" | {val:<10.4f}"
                else:
                    row_str += f" | {val:<10}"
            print(row_str)


if __name__ == "__main__":
    main()
