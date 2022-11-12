import logging
import time
from collections import deque
from copy import deepcopy

import numpy as np
import pyaudio
from numpy.typing import NDArray
from scipy.signal import find_peaks

PYAUDIO_TO_NUMPY_FORMAT = {
    pyaudio.paInt8: np.uint8,
    pyaudio.paInt16: np.uint16,
    pyaudio.paInt32: np.uint32,
}

MAX_NOTABLE_FREQUENCIES = 4

MAX_ECHO_BUFFER = 1000


def check_status(status):
    if status == 4:
        print("UNDERRUN!")
        return
    assert status == 0, status


def sort_notable(ix, y):
    # Sort in decreasing order
    sorted_ix = np.argsort(-y)
    return (
        ix[sorted_ix][:MAX_NOTABLE_FREQUENCIES],
        y[sorted_ix][:MAX_NOTABLE_FREQUENCIES],
    )


class AudioInput:
    def __init__(self):
        self.frames_per_buffer = 512

        self.pyaudio_obj = pyaudio.PyAudio()

        self.print_all_device_info()

        self.rate = int(
            self.pyaudio_obj.get_default_input_device_info()["defaultSampleRate"]
        )

        self.input_buffer = None
        self.output_buffer = None

        # TODO fix 8 bit
        self.format = pyaudio.paInt32
        self.np_format = PYAUDIO_TO_NUMPY_FORMAT[self.format]

        logging.info(
            "Device info: %s", self.pyaudio_obj.get_default_input_device_info()
        )

        new_rate = self.rate
        # TODO adjustable rate factor (max Hz of input...)
        # try 2, 4, 0.5...
        self.in_stream = self.pyaudio_obj.open(
            input=True,
            channels=1,
            # TODO configurable, None is default
            input_device_index=None,
            rate=self.rate,
            format=self.format,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self.receive_audio,
        )
        logging.info(
            "Input: pyaudio format=%s, %sHz, frames per buffer=%s",
            self.format,
            self.rate,
            self.frames_per_buffer,
        )
        logging.info("Input latency: %s", self.in_stream.get_input_latency())

        # Whether to echo the input to the output device
        self.echo = True

        # Don't play until X seconds have gone by...
        self.echo_delay_seconds = 0
        self.echo_buffer = deque()

        self.output_frame_count = None
        self.output_time = None
        self.input_time = None

        if self.echo:
            self.out_stream = self.pyaudio_obj.open(
                output=True,
                # TODO configurable
                output_device_index=None,
                channels=1,
                rate=self.rate,
                format=self.format,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self.send_audio,
            )
            logging.info("Output device opened")
            logging.info(
                "Output device latency: %s", self.out_stream.get_output_latency()
            )
        else:
            self.out_stream = None

        self.x: NDArray[np.float64] = self.filter(
            np.fft.rfftfreq(self.frames_per_buffer, 1 / self.rate)
        )
        self.max_ix = len(self.x)
        self.max_x = max(self.x)
        self.min_x = min(self.x)
        self.y: NDArray[np.float64] = np.random.rand(self.max_ix)
        self.max_y = 1
        self.min_y = 0.25

    def print_all_device_info(self):
        for device_index in range(self.pyaudio_obj.get_device_count()):
            logging.info(
                "Info for device %s\n%s",
                device_index,
                self.pyaudio_obj.get_device_info_by_index(device_index),
            )

    def receive_audio(self, in_data, frame_count, time_info, status):
        self.input_time = time_info["current_time"]
        self.input_frame_count = frame_count

        # THIS FUNCTION MUST BE VERY FAST
        # TODO add option to ignore exceptions
        # TODO ignore status==4, underrun error
        assert frame_count == self.frames_per_buffer, (
            frame_count,
            self.frames_per_buffer,
        )
        check_status(status)

        self.input_buffer = in_data

        # TODO adjust MAX_ECHO_BUFFER bsaed on echo time
        self.output_buffer = deepcopy(self.input_buffer)
        if self.echo and len(self.echo_buffer) < MAX_ECHO_BUFFER:
            self.echo_buffer.append((self.input_time, self.output_buffer))

        return (None, pyaudio.paContinue)

    def send_audio(self, in_data, frame_count, time_info, status):
        self.output_frame_count = frame_count
        self.output_time = time_info["current_time"]
        assert frame_count == self.frames_per_buffer, (
            frame_count,
            self.frames_per_buffer,
        )
        check_status(status)

        if self.output_buffer is not None:
            output_buffer = self.output_buffer
        else:
            output_buffer = bytes([0] * frame_count)
        return (output_buffer, pyaudio.paContinue)

    def filter(self, array):
        # The first element of the FFT is huge, much greater than MAXIMUM_FFT_MAGNITUDE
        # [0] is a special value
        # [-1] is also a special value
        # Filter both out
        return array[1:-1]

    def update_y(self):
        if self.input_buffer is not None:
            array = np.frombuffer(self.input_buffer, dtype=self.np_format)
            self.y = self.filter(np.abs(np.fft.rfft(array)))
            self.max_y = max(self.y)

        logging.debug(
            "I/O times: %s %s %s/%s/%s",
            time.time() - self.input_time if self.input_time else self.input_time,
            time.time() - self.output_time if self.output_time else self.output_time,
            self.output_frame_count,
            self.input_frame_count,
            self.frames_per_buffer,
        )

        if self.echo:
            if self.echo_delay_seconds > 0:
                oldest_buffer_age = -1
                if len(self.echo_buffer) > 0:
                    buffer_time, buffer = self.echo_buffer[0]
                    oldest_buffer_age = time.time() - buffer_time

                logging.debug(
                    "Echo buffer: age=%s len=%s",
                    oldest_buffer_age,
                    len(self.echo_buffer),
                )

                if oldest_buffer_age < self.echo_delay_seconds:
                    self.output_buffer = None
                else:
                    _, self.output_buffer = self.echo_buffer.popleft()

    def run(self):
        self.in_stream.start_stream()
        if self.out_stream is not None:
            self.out_stream.start_stream()

    def shutdown(self):
        logging.info("Closing audio stream(s)")
        self.in_stream.stop_stream()
        self.in_stream.close()
        logging.info("Input audio stream closed")
        if self.out_stream is not None:
            self.out_stream.stop_stream()
            self.out_stream.close()
            logging.info("Output audio stream closed")

    def top_magnitudes(self):
        top = np.argpartition(-self.y, MAX_NOTABLE_FREQUENCIES)[
            :MAX_NOTABLE_FREQUENCIES
        ]
        return sort_notable(top, self.y[top])

    def peaks(self):
        top, _ = find_peaks(
            self.y, height=self.min_y * self.max_y
        )  # , distance=len(self.x) // )
        return sort_notable(top, self.y[top])
