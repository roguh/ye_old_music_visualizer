import logging

import numpy as np
import pyaudio
from scipy.signal import find_peaks
from numpy.typing import NDArray

PYAUDIO_TO_NUMPY_FORMAT = {pyaudio.paInt16: np.uint16}

MAX_NOTABLE_FREQUENCIES = 4


def sort_notable(ix, y):
    # Sort in decreasing order
    sorted_ix = np.argsort(-y)
    return (
        ix[sorted_ix][:MAX_NOTABLE_FREQUENCIES],
        y[sorted_ix][:MAX_NOTABLE_FREQUENCIES],
    )


class AudioInput:
    def __init__(self):
        self.frames_per_buffer = 1024

        self.pyaudio_obj = pyaudio.PyAudio()

        self.rate = int(
            self.pyaudio_obj.get_default_input_device_info()["defaultSampleRate"]
        )

        self.buffer = None
        self.x: NDArray[np.float64] = self.filter(
            np.fft.rfftfreq(self.frames_per_buffer, 1 / self.rate)
        )
        self.y: NDArray[np.float64] = np.random.rand(len(self.x))
        self.max_y = 1

        self.format = pyaudio.paInt16
        self.np_format = PYAUDIO_TO_NUMPY_FORMAT[self.format]

        self.stream = self.pyaudio_obj.open(
            input=True,
            channels=1,
            rate=self.rate,
            format=self.format,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self.receive_audio,
        )

    def receive_audio(self, in_data, frame_count, time_info, status):
        # THIS FUNCTION MUST BE VERY FAST
        # TODO add option to ignore exceptions
        assert frame_count == self.frames_per_buffer, (
            frame_count,
            self.frames_per_buffer,
        )
        assert status == 0, status

        self.buffer = in_data
        return (None, pyaudio.paContinue)

    def filter(self, array):
        # For some reason, the first element of the FFT is huge
        # much greater than MAXIMUM_FFT_MAGNITUDE
        return array[1:]

    def update_y(self):
        if self.buffer is None:
            return

        array = np.frombuffer(self.buffer, dtype=self.np_format)
        self.y = self.filter(np.abs(np.fft.rfft(array)))

        self.max_y = max(self.y)

    def run(self):
        self.stream.start_stream()

    def shutdown(self):
        logging.info("Closing audio stream")
        self.stream.stop_stream()
        self.stream.close()

    def top_magnitudes(self):
        top = np.argpartition(-self.y, MAX_NOTABLE_FREQUENCIES)[
            :MAX_NOTABLE_FREQUENCIES
        ]
        return sort_notable(top, self.y[top])

    def peaks(self):
        top, _ = find_peaks(
            self.y, height=self.max_y * 0.05  # , distance=len(self.x) // 20
        )
        return sort_notable(top, self.y[top])
