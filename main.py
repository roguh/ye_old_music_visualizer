#!/usr/bin/env python3
import logging

import visualization
import audio_input

logging.basicConfig(
    format="%(levelname)s %(asctime)s: %(message)s", level=logging.DEBUG
)


if __name__ == "__main__":
    visualization.run(audio_input.AudioInput())
