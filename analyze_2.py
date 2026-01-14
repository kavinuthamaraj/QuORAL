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

parser = argparse.ArgumentParser(description='APODAS Analysis for Photon Count Traces')
parser.add_argument('--data', type=str, required=True, help='Path to data folder with APODAS log files')
parser.add_argument('--threads', type=int, required=True, help='Multiprocessing threads')
parser.add_argument('--bin_width', type=float, default=0.0001, help='Time bin width in seconds (e.g., 0.0001 = 0.1 ms for fine resolution)')
parser.add_argument('--show_rate', action='store_true', help='Plot photon rate (Hz) instead of raw counts')
parser.add_argument('--interactive', action='store_true', help='Show interactive plot window')
parser.add_argument('--zoom_low_rate', action='store_true', help='Zoom y-axis for low-rate regions (e.g., single-atom signals)')
parser.add_argument('--plot_per_file', action='store_true', help='Generate separate plots for each log file (useful for short gates)')
args = parser.parse_args()

if __name__ == '__main__':
    archive_path = './archives/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    Path(archive_path).mkdir(parents=True, exist_ok=True)

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

    raw_data_files = sorted(glob.glob(os.path.join(args.data, 'Apodas*.log')))
    logging.info(f"Found {len(raw_data_files)} raw data files")

    PAT = photon_arrival_timings.Photon_Arrival_Timings(archive_path)
    for file in raw_data_files:
        PAT.retrieve_from_log_file(file)

    timings_files = sorted(glob.glob(os.path.join(archive_path, 'APD_A_detection_timimngs*.dat')))
    logging.info(f"Found {len(timings_files)} timings files for APD_A")

    all_ts_ns = []
    file_ts_dict = {}  
    for f in timings_files:
        try:
            ts_ns = np.loadtxt(f, usecols=2)
            if len(ts_ns) > 0:
                all_ts_ns.append(ts_ns)
                file_ts_dict[f] = ts_ns
                logging.info(f"Loaded {len(ts_ns)} timestamps from {os.path.basename(f)} (max time: {ts_ns.max()/1e9:.3f} s)")
            else:
                logging.warning(f"No data in {f}")
        except Exception as e:
            logging.warning(f"Skipping {f}: {e}")

    if not all_ts_ns:
        raise ValueError("No timestamps found! Check files or setup.")

    all_ts = np.sort(np.concatenate(all_ts_ns)) / 1e9
    bins = np.arange(all_ts.min(), all_ts.max() + args.bin_width, args.bin_width)
    counts, bin_edges = np.histogram(all_ts, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(50, 6))  
    values = counts / args.bin_width if args.show_rate else counts
    ylabel = 'Photon Rate (Hz)' if args.show_rate else 'Photon Counts per Bin'
    line_color = 'red' if args.show_rate else 'blue'
    plt.step(bin_centers, values, where='mid', color=line_color, linewidth=0.8, alpha=0.9)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'Full Fluorescence Trace - APD A | Total time: {all_ts.max() - all_ts.min():.2f} s | {len(all_ts)} photons', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    if args.zoom_low_rate:
        plt.ylim(0, 20000)  
    plt.tight_layout()

    plot_path = os.path.join(archive_path, 'full_fluorescence_trace_APD_A.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved full plot: {plot_path}")

    if args.interactive:
        plt.show()
    else:
        plt.close()

    if args.plot_per_file:
        for f, ts_ns in file_ts_dict.items():
            ts = np.sort(ts_ns) / 1e9
            if len(ts) == 0:
                continue
            bins_f = np.arange(ts.min(), ts.max() + args.bin_width, args.bin_width)
            counts_f, edges_f = np.histogram(ts, bins=bins_f)
            centers_f = (edges_f[:-1] + edges_f[1:]) / 2
            plt.figure(figsize=(14, 6))
            values_f = counts_f / args.bin_width if args.show_rate else counts_f
            plt.step(centers_f, values_f, where='mid', color=line_color, linewidth=1.5, alpha=0.9)
            plt.xlabel('Time (seconds)')
            plt.ylabel(ylabel)
            plt.title(f'Per-File Trace: {os.path.basename(f)} | Duration: {ts.max() - ts.min():.2f} s | {len(ts)} photons')
            plt.grid(True, alpha=0.3, linestyle='--')
            if args.zoom_low_rate:
                plt.ylim(0, 20000)
            plt.tight_layout()
            file_plot_path = os.path.join(archive_path, f'file_trace_{os.path.basename(f)}.png')
            plt.savefig(file_plot_path, dpi=300)
            logging.info(f"Saved per-file plot: {file_plot_path}")
            plt.close()
