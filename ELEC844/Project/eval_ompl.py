import csv
import sys
import os
import statistics
import math


def is_valid(value):
    """
    Returns the float value if valid, otherwise returns None.
    Filters: 0, -1, inf, and the 32-bit float max sentinel.
    """
    try:
        f_val = float(value)

        # 1. Ignore 0 and -1
        if f_val == 0 or f_val == -1:
            return None

        # 2. Ignore actual Infinity
        if math.isinf(f_val):
            return None

        # 3. Ignore the 32-bit float max sentinel (3.4028e+38)
        # We use a threshold of 1e30 to catch any similar "placeholder" values.
        if abs(f_val) > 1e30:
            return None

        return f_val
    except (ValueError, TypeError):
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python calc_medians.py <filename.csv>")
        return

    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found.")
        return

    column_data = {}

    try:
        with open(filename, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            for h in headers:
                column_data[h] = []

            for row in reader:
                for h in headers:
                    val = is_valid(row[h])
                    if val is not None:
                        column_data[h].append(val)

        # Print Table
        header_text = f"{'Column Name':<30} | {
            'Median':<12} | {'Mean':<12} | {'Count'}"
        print(header_text)
        print("-" * len(header_text))

        for h in headers:
            data = column_data[h]
            if data:
                med = statistics.median(data)
                mean = statistics.mean(data)
                print(f"{h:<30} | {med:<12.4f} | {mean:<12.4f} | {len(data)}")
            else:
                print(f"{h:<30} | {'N/A':<12} | 0")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
