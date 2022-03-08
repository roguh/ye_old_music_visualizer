import logging
from collections import deque
from typing import Tuple

import numpy as np
import palettable
import pyglet

import audio_input

loop = pyglet.app.EventLoop()
window = pyglet.window.Window(800, 800, "FFT")

Color = Tuple[int, int, int]


def freq_to_color(hertz: float, max_hertz: float) -> Color:
    return tuple(
        int(c) for c in (255 * hertz / max_hertz, 0, 255 - 128 * hertz / max_hertz)
    )


def notable_freq_to_color(order: int):
    return getattr(
        palettable.colorbrewer.diverging,
        "RdBu_" + str(min(audio_input.MAX_NOTABLE_FREQUENCIES, 11)),
    ).colors[order]


def magnitude_to_color(mag: float, max_mag: float) -> Color:
    return tuple(
        255 if abs(c) == float("inf") else int(c)
        for c in (
            128 * mag / max_mag,
            0,
            128 - 128 * mag / max_mag,
        )
    )


class AudioVisualization:
    def __init__(self, _audio_input: audio_input.AudioInput):
        self.audio_input = _audio_input

        window.push_handlers(
            self.on_draw,
            self.on_close,
            self.on_resize,
            self.on_key_press,
        )

        self.fps_display = pyglet.window.FPSDisplay(window)
        self.visualizations = [
            vis(self.audio_input) for vis in [BarVisualization, Fancy]
        ]
        self.vis = 1

    def next_vis(self):
        self.vis = (self.vis + 1) % len(self.visualizations)

    def prev_vis(self):
        self.vis -= 1
        if self.vis < 0:
            self.vis = len(self.visualizations) - 1

    def update(self, dt):
        self.audio_input.update_y()
        for vis in self.visualizations:
            vis.update(dt)

    def on_draw(self):
        window.clear()
        self.fps_display.draw()
        self.visualizations[self.vis].on_draw()

    def shutdown(self):
        self.audio_input.shutdown()
        loop.exit()
        print("Goodbye!")

    def on_close(self):
        logging.info("Window is closing")
        self.shutdown()

    def on_resize(self, width, height):
        logging.debug("Resize W: %s H: %s", width, height)

        self.visualizations[self.vis].on_resize(width, height)

    def on_key_press(self, symbol, modifiers):
        logging.debug("Key press: %s %s", symbol, modifiers)

        if symbol == pyglet.window.key.ESCAPE:
            window.has_exit = True

        if symbol == pyglet.window.key.LEFT:
            self.next_vis()

        if symbol == pyglet.window.key.RIGHT:
            self.prev_vis()

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)
        self.audio_input.run()
        loop.run()


class BarVisualization:
    def __init__(self, _audio_input: audio_input.AudioInput):
        self.audio_input = _audio_input

        self.batch = pyglet.graphics.Batch()
        self.labels = pyglet.graphics.Batch()

        self.bars = [
            pyglet.shapes.Rectangle(
                x=0,
                y=0,
                color=freq_to_color(hz, max(self.audio_input.x)),
                width=1,
                height=1,
                batch=self.batch,
            )
            for hz in self.audio_input.x
        ]

        self.notable_frequencies = [
            pyglet.shapes.Circle(
                x=0,
                y=0,
                color=(255, 255, 128),
                radius=10,
                batch=self.batch,
            )
            for _ in range(audio_input.MAX_NOTABLE_FREQUENCIES)
        ]

        self.label = pyglet.text.Label(
            "FFT",
            font_name="Times New Roman",
            font_size=36,
            x=0,
            y=0,
            anchor_x="center",
            anchor_y="center",
            batch=self.labels,
        )

    def map_y(self, y):
        return 0.9 * window.height * y / self.audio_input.max_y

    def map_ix(self, ix):
        return window.width * ix / len(self.audio_input.x)

    def update_bars(self):
        width = window.width / len(self.audio_input.x)
        for ix, (rect, y) in enumerate(zip(self.bars, self.audio_input.y)):
            rect.height = self.map_y(y)
            rect.x = self.map_ix(ix)
            rect.width = width

    def update_notable_frequencies(self):
        ixs, ys = self.audio_input.peaks()

        freq_count = 0
        for freq_count, (ix, y, circle) in enumerate(
            zip(ixs, ys, self.notable_frequencies)
        ):
            circle.x = self.map_ix(ix)
            circle.y = self.map_y(y)
            circle.radius = max(5, min(window.width, window.height) * 0.01)
            circle.color = notable_freq_to_color(freq_count)

        for circle in self.notable_frequencies[freq_count:]:
            circle.x = -circle.radius * 2

    def update_labels(self):
        self.label.x = window.width // 2
        self.label.y = window.height - self.label.font_size

    def update(self, dt):
        self.update_bars()
        self.update_notable_frequencies()
        self.update_labels()

    def on_resize(self, width, height):
        pass

    def on_draw(self):
        self.batch.draw()
        self.labels.draw()


class Fancy:
    def __init__(self, _audio_input: audio_input.AudioInput):
        self.audio_input = _audio_input
        self.batches = [
            pyglet.graphics.Batch() for _ in range(audio_input.MAX_NOTABLE_FREQUENCIES)
        ]

        self.max_rows = len(self.audio_input.x) // 3
        self.max_ix = len(self.audio_input.x)

        self.rows = deque()

    def get_square_height(self, ix):
        return window.height / self.max_ix * np.log(self.max_ix / (ix + 1))

    def get_square_width(self):
        return np.log(self.max_ix) / 4 * window.width / self.max_ix

    def get_row(self, peak_ixs):
        squares = []
        # pixels = [
        #    pyglet.shapes.Rectangle(
        #        x=self.map_ix(ix),
        #        y=0,
        #        height=self.get_square_height(),
        #        width=self.get_square_width(),
        #        color=(255, 255, 255),
        #        batch=self.batch,
        #    )
        #    for ix in range(len(self.audio_input.x))
        # ]

        # for y, pixel in zip(self.audio_input.y, pixels):
        #    pixel.color = magnitude_to_color(np.log(y), np.log(self.audio_input.max_y))

        for order, ix in enumerate(peak_ixs):
            square = pyglet.shapes.Rectangle(
                y=self.map_ix(ix),
                x=self.map_y(len(self.rows), ix),
                height=self.get_square_height(ix),
                width=self.get_square_width(),
                color=notable_freq_to_color(order),
                batch=self.batches[order],
            )
            squares.insert(0, square)

        return squares

    def update_squares(self):
        for y, row in enumerate(self.rows):
            for ix, square in enumerate(row):
                square.x = self.map_y(y, ix)
                square.height = self.get_square_height(ix)
                square.width = self.get_square_width()

    def map_ix(self, ix):
        return window.width * ix / len(self.audio_input.x)

    def map_y(self, y, ix):
        return window.height / self.max_rows * y

    def update(self, dt):
        peak_ixs, _ = self.audio_input.peaks()
        self.rows.append(self.get_row(peak_ixs))
        if len(self.rows) > self.max_rows:
            self.rows.popleft()
        self.update_squares()

    def on_resize(self, width, height):
        self.update_squares()

    def on_draw(self):
        for batch in reversed(self.batches):
            batch.draw()


def run(audio_input_instance):
    vis = AudioVisualization(audio_input_instance)
    try:
        vis.run()
    except KeyboardInterrupt:
        vis.shutdown()
