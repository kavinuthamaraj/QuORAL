import os
from pathlib import Path
import logging

class Photon_Arrival_Timings:
    def __init__(self, archive_path):
        self.apd_a_detection_prefix = "APD_A_detection_timimngs"
        self.apd_b_detection_prefix = "APD_B_detection_timimngs"
        self.packet_counter_prefix = "Packet_counter"
        self.apd_coincident_prefix = "APD_coincident_detection_timimngs"
        self.apd_a_detection = 0
        self.apd_b_detection = 0
        self.coincidence_detection = 0
        self.clock_period_ns = 5
        self.time_for_one_wrap_around = 160
        self.number_of_wrap_around = 0
        self.sub_data_array_size = 1024
        self.packet_header_size = 90
        self.first_wrap_around = 0
        self.last_udp_packet_counter = None  # ← Key change: persist across files
        self.archive_path = archive_path

    def make_file_name(self, prefix, file):
        return f"{prefix}_{Path(file).with_suffix('.dat')}"

    def make_file_path(self, prefix, file):
        return os.path.join(self.archive_path, self.make_file_name(prefix, os.path.basename(file)))

    def retrieve_from_log_file(self, file):
        logging.info(f"Retrieving photon arrival timings from file {file}")
        self.fp_apd_a_detection = open(self.make_file_path(self.apd_a_detection_prefix, file), 'w')
        self.fp_apd_b_detection = open(self.make_file_path(self.apd_b_detection_prefix, file), 'w')
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
                        self.last_udp_packet_counter = udp_packet_counter  # ← Update for next file too

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
        self.fp_apd_b_detection.close()
        self.fp_apd_coincidence.close()
        self.fp_packet_counter.close()

    def detection_and_timing_of_pulses(self, sub_data_array, packet_counter):
        time_of_occurrence_apd_a = time_of_occurrence_apd_b = time_of_occurrence_coincidence = 0
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
                    elif 64 <= sub_data_array[i] <= 95:
                        self.apd_b_detection += 1
                        time_of_occurrence_apd_b = (sub_data_array[i] - 64) * self.clock_period_ns + (self.number_of_wrap_around * self.time_for_one_wrap_around)
                        self.fp_apd_b_detection.write(f"{packet_counter}\t{self.apd_b_detection}\t{time_of_occurrence_apd_b}\n")
                    elif 96 <= sub_data_array[i] <= 127:
                        self.coincidence_detection += 1
                        self.apd_a_detection += 1
                        self.apd_b_detection += 1
                        time_of_occurrence_coincidence = (sub_data_array[i] - 96) * self.clock_period_ns + (self.number_of_wrap_around * self.time_for_one_wrap_around)
                        self.fp_apd_coincidence.write(f"{packet_counter}\t{self.coincidence_detection}\t{time_of_occurrence_coincidence}\n")
                        self.fp_apd_a_detection.write(f"{packet_counter}\t{self.apd_a_detection}\t{time_of_occurrence_coincidence}\n")
                        self.fp_apd_b_detection.write(f"{packet_counter}\t{self.apd_b_detection}\t{time_of_occurrence_coincidence}\n")
                else:
                    if sub_data_array[i] == 159:
                        self.number_of_wrap_around += 1
                    elif 160 <= sub_data_array[i] <= 191:
                        self.apd_a_detection += 1
                        time_of_occurrence_apd_a = (sub_data_array[i] - 160) * self.clock_period_ns + (self.number_of_wrap_around * self.time_for_one_wrap_around)
                        self.number_of_wrap_around += 1
                        self.fp_apd_a_detection.write(f"{packet_counter}\t{self.apd_a_detection}\t{time_of_occurrence_apd_a}\n")
                    elif 192 <= sub_data_array[i] <= 223:
                        self.apd_b_detection += 1
                        time_of_occurrence_apd_b = (sub_data_array[i] - 192) * self.clock_period_ns + (self.number_of_wrap_around * self.time_for_one_wrap_around)
                        self.number_of_wrap_around += 1
                        self.fp_apd_b_detection.write(f"{packet_counter}\t{self.apd_b_detection}\t{time_of_occurrence_apd_b}\n")
                    elif 224 <= sub_data_array[i] <= 255:
                        self.coincidence_detection += 1
                        self.apd_a_detection += 1
                        self.apd_b_detection += 1
                        time_of_occurrence_coincidence = (sub_data_array[i] - 224) * self.clock_period_ns + (self.number_of_wrap_around * self.time_for_one_wrap_around)
                        self.number_of_wrap_around += 1
                        self.fp_apd_coincidence.write(f"{packet_counter}\t{self.coincidence_detection}\t{time_of_occurrence_coincidence}\n")
                        self.fp_apd_a_detection.write(f"{packet_counter}\t{self.apd_a_detection}\t{time_of_occurrence_coincidence}\n")
                        self.fp_apd_b_detection.write(f"{packet_counter}\t{self.apd_b_detection}\t{time_of_occurrence_coincidence}\n")
