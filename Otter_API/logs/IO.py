import csv
import os

import datetime
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_LOG_DIR = f"logs/sim_logs/{ts}"

def log_to_csv(simTime, simData, targetData, filename, verbose=False):
    if not verbose:
        return

    # Prepend user-chosen filename with the base log directory
    full_path = os.path.join(BASE_LOG_DIR, filename)

    # Ensure folder exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    header = ["simTime"]
    header += [f"simData_{i}" for i in range(len(simData[0]))]
    header += [f"targetData_{i}" for i in range(len(targetData[0]))]

    with open(full_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for t, s, tg in zip(simTime, simData, targetData):
            writer.writerow([t] + list(s) + list(tg))

    print(f"CSV log saved to {full_path}")

def log_params(params: dict, filename="run_parameters.txt", verbose=False):
    if not verbose:
        return

    full_path = os.path.join(BASE_LOG_DIR, filename)


    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    print(f"Parameter log saved to {full_path}")