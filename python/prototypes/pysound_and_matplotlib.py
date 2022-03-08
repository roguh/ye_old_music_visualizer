import logging
import time
from scipy.signal import find_peaks

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

MAXIMUM_FFT_MAGNITUDE = 4e7

DEFAULT_FRAMES_PER_BUFFER = 1024

PYAUDIO_TO_NUMPY_FORMAT = {pyaudio.paInt16: np.uint16}

pyaudio_obj = pyaudio.PyAudio()

DEFAULT_RATE = int(pyaudio_obj.get_default_input_device_info()["defaultSampleRate"])


def record(
    rate,
    frames_per_buffer=DEFAULT_FRAMES_PER_BUFFER,
    frame_count=None,
    seconds=1,
    channels=1,
    format=pyaudio.paInt16,
):

    frames = []

    stream = pyaudio_obj.open(
        format=format,
        channels=channels,
        rate=DEFAULT_RATE,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )

    if frame_count is None:
        frame_count = int(rate / frames_per_buffer * seconds)

    for _ in range(0, frame_count):
        data = stream.read(frames_per_buffer)
        frames.append(data)

    return frames


def fft(seconds=1, frame_count=None):
    # https://realpython.com/python-scipy-fft
    frames_per_buffer = DEFAULT_FRAMES_PER_BUFFER
    format = pyaudio.paInt16
    buffers = record(
        seconds=seconds,
        frame_count=frame_count,
        frames_per_buffer=frames_per_buffer,
        rate=DEFAULT_RATE,
        format=format,
    )
    arrays = [
        np.frombuffer(buffer, dtype=PYAUDIO_TO_NUMPY_FORMAT[format])
        for buffer in buffers
    ]
    frequencies = np.fft.rfftfreq(frames_per_buffer, 1 / DEFAULT_RATE)
    fft = np.fft.rfft(arrays)
    return frequencies, np.abs(fft)


def animate(x, ys, seconds=1):

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, ys[0])

    def animate(y):
        line.set_ydata(y)
        return (line,)

    return animation.FuncAnimation(
        fig, animate, frames=ys, interval=seconds * 1000 / len(ys)
    )


def realtimefft(buffer: bytes, format):
    # buffer = stream.read(DEFAULT_FRAMES_PER_BUFFER)
    array = np.frombuffer(buffer, dtype=PYAUDIO_TO_NUMPY_FORMAT[format])
    fft_abs = np.abs(np.fft.rfft(array))
    return fft_abs


def ffilter(data):
    # For some reason, the first element of the FFT is huge
    # much greater than MAXIMUM_FFT_MAGNITUDE
    return data[1:]


def realtime_plot():
    # THE AUDIO DATA
    buffer = None

    format = pyaudio.paInt16

    def receive_audio(in_data, frame_count, time_info, status):
        # THIS FUNCTION MUST BE VERY FAST
        # TODO add option to ignore exceptions
        assert frame_count == DEFAULT_FRAMES_PER_BUFFER, (
            frame_count,
            DEFAULT_FRAMES_PER_BUFFER,
        )
        assert status == 0, status

        nonlocal buffer
        buffer = in_data
        logging.debug("Audio received")
        return (None, pyaudio.paContinue)

    stream = pyaudio_obj.open(
        format=format,
        channels=1,
        rate=DEFAULT_RATE,
        input=True,
        output=False,
        frames_per_buffer=DEFAULT_FRAMES_PER_BUFFER,
        stream_callback=receive_audio,
    )

    stream.start_stream()

    x = ffilter(np.fft.rfftfreq(DEFAULT_FRAMES_PER_BUFFER, 1 / DEFAULT_RATE))

    y = np.zeros(len(x))

    matplotlib_closed = False

    plt.ion()

    figure, ax = plt.subplots(figsize=(8, 6))

    def on_close(_):
        nonlocal matplotlib_closed
        matplotlib_closed = True
        logging.info("Matplotlib figure has been closed")

    figure.canvas.mpl_connect("close_event", on_close)

    (line,) = ax.plot(x, y)
    (line2,) = ax.plot([], [], "bo")

    plt.ylim(0, MAXIMUM_FFT_MAGNITUDE)

    plt.title("FFT", fontsize=25)

    plt.xlabel("Hz", fontsize=18)
    plt.ylabel("magnitude", fontsize=18)

    try:
        while not matplotlib_closed and stream.is_active():
            if buffer is None:
                continue
            y = ffilter(realtimefft(buffer, format))

            line.set_ydata(y)

            # Find the top 4 values
            # top = np.argpartition(-y, 4)[:4]

            # Find local maxima
            top, _ = find_peaks(y, height=1e6, distance=len(x) // 20)
            line2.set_ydata(y[top])
            line2.set_xdata(x[top])

            figure.canvas.draw()
            figure.canvas.flush_events()

            logging.debug("Updated plot")
    except KeyboardInterrupt:
        if not matplotlib_closed:
            plt.close()
        stream.stop_stream()
        stream.close()

    logging.info("STOPPED")

    return x, y


if __name__ == "__main__":
    realtime_plot()
