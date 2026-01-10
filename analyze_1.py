import argparse
import glob
import os
import logging
from colorlog import ColoredFormatter
from datetime import datetime
from pathlib import Path
import photon_arrival_timings
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Argument Parser -------------------
parser = argparse.ArgumentParser(description='APODAS Analysis for Photon Count Traces')
parser.add_argument('--data', type=str, required=True, help='Path to data folder with APODAS log files')
parser.add_argument('--threads', type=int, required=True, help='Multiprocessing threads')  # kept for future use
parser.add_argument('--bin_width', type=float, default=0.002, help='Time bin width in seconds (e.g., 0.002 = 2 ms)')
parser.add_argument('--show_rate', action='store_true', help='Plot photon rate (Hz) instead of raw counts')
parser.add_argument('--detector', type=str, default='A', choices=['A', 'B'], help='Detector to plot (A or B)')
parser.add_argument('--interactive', action='store_true', help='Show interactive plot window')
args = parser.parse_args()

# ------------------- Main Execution -------------------
if __name__ == '__main__':
    archive_path = './archives/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    Path(archive_path).mkdir(parents=True, exist_ok=True)

    # Logging setup
    LOGFORMAT = "%(asctime)s.%(msecs)03d  %(log_color)s%(levelname)-8s%(reset)s %(module)s - %(funcName)s: %(log_color)s%(message)s%(reset)s"
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(archive_path, "logs.log")),
            stream
        ]
    )
    logging.info(f"Analyzing APODAS data from dir={args.data} with archive path={archive_path}")

    # Find and sort log files (chronological order)
    raw_data_files = sorted(glob.glob(os.path.join(args.data, 'Apodas*.log')))
    logging.info(f"Found {len(raw_data_files)} raw data files")

    # Create single PAT instance and process files SEQUENTIALLY
    PAT = photon_arrival_timings.Photon_Arrival_Timings(archive_path)
    for file in raw_data_files:
        PAT.retrieve_from_log_file(file)

    # Load all extracted timestamps for the chosen detector
    if args.detector == 'A':
        timings_files = sorted(glob.glob(os.path.join(archive_path, 'APD_A_detection_timimngs*.dat')))
    else:
        timings_files = sorted(glob.glob(os.path.join(archive_path, 'APD_B_detection_timimngs*.dat')))

    logging.info(f"Found {len(timings_files)} timings files for APD_{args.detector}")

    all_ts_ns = []
    for f in timings_files:
        try:
            ts_ns = np.loadtxt(f, usecols=2)
            if len(ts_ns) > 0:
                all_ts_ns.append(ts_ns)
                logging.info(f"Loaded {len(ts_ns)} timestamps from {os.path.basename(f)} (max time: {ts_ns.max()/1e9:.3f} s)")
        except Exception as e:
            logging.warning(f"Skipping {f}: {e}")

    if not all_ts_ns:
        raise ValueError("No timestamps found! Check files or detector choice.")

    # Combine and convert to seconds
    all_ts = np.concatenate(all_ts_ns) / 1e9
    all_ts = np.sort(all_ts)

    # Histogram / binning
    bins = np.arange(all_ts.min(), all_ts.max() + args.bin_width, args.bin_width)
    counts, bin_edges = np.histogram(all_ts, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot - clean line style
    plt.figure(figsize=(14, 6))
    if args.show_rate:
        values = counts / args.bin_width
        ylabel = 'Photon Rate (Hz)'
        line_color = 'red'
    else:
        values = counts
        ylabel = 'Photon Counts per Bin'
        line_color = 'blue'

    plt.step(bin_centers, values, where='mid', color=line_color, linewidth=1.5, alpha=0.9)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'Fluorescence Trace - APD {args.detector} | Total time: {all_ts.max() - all_ts.min():.2f} s | {len(all_ts)} photons', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save high-quality PNG
    plot_path = os.path.join(archive_path, f'fluorescence_trace_APD_{args.detector}.png')
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    logging.info(f"Saved plot as PNG: {plot_path}")

    if args.interactive:
        plt.show()
    else:
        plt.close()
