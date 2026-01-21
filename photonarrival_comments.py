import os
from pathlib import Path
import logging

class Photon_Arrival_Timings:
    def __init__(self, archive_path):
        """
        This is where we set up everything the class needs. We define the file prefixes we'll use for saving the extracted data,
        reset all counters (photon detections, coincidences, wrap-arounds), and remember the hardware timing constants.
        The archive_path is passed in so all output files go to the correct timestamped folder. We also keep track of the last
        UDP packet counter across files so we can detect missed packets even between different log files.
        """
        self.apd_a_detection_prefix = "APD_A_detection_timimngs"
        self.apd_coincident_prefix = "APD_coincident_detection_timimngs"
        self.packet_counter_prefix = "Packet_counter"
        self.apd_a_detection = 0
        self.coincidence_detection = 0
        self.clock_period_ns = 5
        self.time_for_one_wrap_around = 160
        self.number_of_wrap_around = 0
        self.sub_data_array_size = 1024
        self.packet_header_size = 90
        self.first_wrap_around = 0
        self.last_udp_packet_counter = None
        self.archive_path = archive_path

    def make_file_name(self, prefix, file):
        """
        Helper to create consistent output filenames. It takes the prefix (like APD_A_detection_timimngs) and appends the original
        log filename without the .log extension, then adds .dat. This keeps everything traceable back to which Apodas_XXXX.log
        file the data came from.
        """
        return f"{prefix}_{Path(file).with_suffix('.dat')}"

    def make_file_path(self, prefix, file):
        """
        Builds the full path for output files by joining the archive folder with the filename we just made. This ensures all
        outputs land in the correct timestamped archive directory.
        """
        return os.path.join(self.archive_path, self.make_file_name(prefix, os.path.basename(file)))

    def retrieve_from_log_file(self, file):
        """
        This is the main workhorse function. For each raw Apodas_*.log file:
        - We open the three output files (APD A, coincidence, packet counter) in write mode.
        - Skip the first 24 bytes of the log (header we don't need).
        - Read the 90-byte packet header, check if it's a valid 1052-byte packet (90 header + 1024 data).
        - Extract the UDP packet counter and check for jumps (missed packets) — if jumped, we add the missed time to wrap-around counter.
        - Read the 1024-byte data array and pass it to detection_and_timing_of_pulses.
        - Write packet stats to Packet_counter file.
        - At the end, write summary stats (total missed packets, total observation time, time minus missed).
        - All this happens in a loop until EOF (StopIteration).
        Important: last_udp_packet_counter is persistent across files, so we catch misses between logs.
        """
        logging.info(f"Retrieving photon arrival timings from file {file}")
        self.fp_apd_a_detection = open(self.make_file_path(self.apd_a_detection_prefix, file), 'w')
        self.fp_apd_coincidence = open(self.make_file_path(self.apd_coincident_prefix, file), 'w')
        self.fp_packet_counter = open(self.make_file_path(self.packet_counter_prefix, file), 'w')

        with open(file, 'rb') as fp_file:
            fp_file.seek(24, 0)

            packet_header = [0] * self.packet_header_size
            packet_counter = 1
            difference = None
            total_missed_packets = 0
            try:
                while True:
                    for i in range(self.packet_header_size):
                        fp_file_chunk = fp_file.read(1)
                        if fp_file_chunk == b'':
                            raise StopIteration
                        packet_header[i] = fp_file_chunk[0]
                    packet_length = packet_header[32] * 256 + packet_header[33]
                    if packet_length == 1052:
                        udp_packet_counter = packet_header[86] * 16777216 + packet_header[87] * 65536 + packet_header[88] * 256 + packet_header[89]
                        if self.last_udp_packet_counter is not None:
                            difference = udp_packet_counter - self.last_udp_packet_counter
                            if difference != 1:
                                logging.warning(f"Non-zero difference, packets missed: {difference - 1} at packet counter: {packet_counter} in file {file}")
                                self.number_of_wrap_around += (difference - 1) * 1024
                                total_missed_packets += difference - 1
                        self.fp_packet_counter.write(f"{packet_counter}\t{udp_packet_counter}\t{difference if difference is not None else 1}\n")
                        self.last_udp_packet_counter = udp_packet_counter

                        sub_data_array = list(fp_file.read(self.sub_data_array_size))
                        self.detection_and_timing_of_pulses(sub_data_array, packet_counter)
                        packet_counter += 1
                    else:
                        logging.warn(f"Invalid packet of length {packet_length} found at seek: {fp_file.tell()} packet counter: {packet_counter - 1}")
                        while True:
                            chunk = list(fp_file.read(4))
                            if chunk == [69, 88, 80, 46]:
                                fp_file.seek(-76, 1)
                                logging.info(f"Found the start of the next packet at seek: {fp_file.tell()}")
                                break
                            elif chunk == []:
                                logging.error("End of file reached before finding the next packet")
                                raise StopIteration
            except StopIteration: pass
            self.fp_packet_counter.write(f"Total missed packets: {total_missed_packets}\n")
            self.fp_packet_counter.write(f"Total time of observation: {self.number_of_wrap_around * self.time_for_one_wrap_around}\n")
            self.fp_packet_counter.write(f"Total time of observation - missed packets time: {(self.number_of_wrap_around - (total_missed_packets * 1024)) * self.time_for_one_wrap_around}\n")
        self.fp_apd_a_detection.close()
        self.fp_apd_coincidence.close()
        self.fp_packet_counter.close()

    def detection_and_timing_of_pulses(self, sub_data_array, packet_counter):
        """
        This is the heart of the parsing logic. For each 1024-byte data chunk from a packet:
        - We loop over every byte in the actual chunk (not fixed 1024, to handle partial last packets safely).
        - First byte == 159 is a wrap-around marker (clock overflow) — we increment the wrap counter.
        - Bytes < 128 are normal detections (no overflow):
          - 32–63: APD A detection (subtract 32 to get offset in clock ticks)
          - 96–127: Coincidence (both APD A and B detected)
        - Bytes >= 128 are overflow detections (extra wrap + offset).
        - For each detection, we calculate the absolute timestamp in ns: offset * 5 ns + wrap * 160 ns.
        - Write the result to the corresponding .dat file (packet_counter, detection count, timestamp_ns).
        - We only write when there's a valid detection code — empty or invalid bytes are ignored.
        This way we get precise photon arrival times, handling clock wraps and overflows correctly.
        """
        time_of_occurrence_apd_a = time_of_occurrence_coincidence = 0
        actual_length = len(sub_data_array)
        for i in range(actual_length):
            if sub_data_array[i] == 159 and self.first_wrap_around == 0:
                self.first_wrap_around = 1
                self.number_of_wrap_around = -1
            if self.first_wrap_around:
                if sub_data_array[i] < 128:
                    if 32 <= sub_data_array[i] <= 63:
                        self.apd_a_detection += 1
                        time_of_occurrence_apd_a = (sub_data_array[i] - 32) * self.clock_period_ns + (self.number_of_wrap_around * self.time_for_one_wrap_around)
                        self.fp_apd_a_detection.write(f"{packet_counter}\t{self.apd_a_detection}\t{time_of_occurrence_apd_a}\n")
                    elif 96 <= sub_data_array[i] <= 127:
                        self.coincidence_detection += 1
                        self.apd_a_detection += 1
                        time_of_occurrence_coincidence = (sub_data_array[i] - 96) * self.clock_period_ns + (self.number_of_wrap_around * self.time_for_one_wrap_around)
                        self.fp_apd_coincidence.write(f"{packet_counter}\t{self.coincidence_detection}\t{time_of_occurrence_coincidence}\n")
                        self.fp_apd_a_detection.write(f"{packet_counter}\t{self.apd_a_detection}\t{time_of_occurrence_coincidence}\n")
                else:
                    if sub_data_array[i] == 159:
                        self.number_of_wrap_around += 1
                    elif 160 <= sub_data_array[i] <= 191:
                        self.apd_a_detection += 1
                        time_of_occurrence_apd_a = (sub_data_array[i] - 160) * self.clock_period_ns + (self.number_of_wrap_around * self.time_for_one_wrap_around)
                        self.number_of_wrap_around += 1
                        self.fp_apd_a_detection.write(f"{packet_counter}\t{self.apd_a_detection}\t{time_of_occurrence_apd_a}\n")
                    elif 224 <= sub_data_array[i] <= 255:
                        self.coincidence_detection += 1
                        self.apd_a_detection += 1
                        time_of_occurrence_coincidence = (sub_data_array[i] - 224) * self.clock_period_ns + (self.number_of_wrap_around * self.time_for_one_wrap_around)
                        self.number_of_wrap_around += 1
                        self.fp_apd_coincidence.write(f"{packet_counter}\t{self.coincidence_detection}\t{time_of_occurrence_coincidence}\n")
                        self.fp_apd_a_detection.write(f"{packet_counter}\t{self.apd_a_detection}\t{time_of_occurrence_coincidence}\n")
