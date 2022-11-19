import atexit


class TriggerDetector:
    """
    Reads predictions and detects activations
    This prevents multiple close activations from occurring when
    the predictions look like ...!!!..!!...
    """
    def __init__(self, chunk_size, threshold=0.5, trigger_level=3):
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.trigger_level = trigger_level
        self.activation = 0

    def update(self, prob):
        # type: (float) -> bool
        """Returns whether the new prediction caused an activation"""
        chunk_activated = prob > self.threshold

        if chunk_activated or self.activation < 0:
            self.activation += 1
            has_activated = self.activation > self.trigger_level
            if has_activated or chunk_activated and self.activation < 0:
                self.activation = -(8 * 2048) // self.chunk_size

            if has_activated:
                return True
        elif self.activation > 0:
            self.activation -= 1
        return False

class PreciseRunner(object):
    """
    Wrapper to use Precise. Example:
    >>> def on_act():
    ...     print('Activation!')
    ...
    >>> p = PreciseRunner(PreciseEngine('./precise-engine'), on_activation=on_act)
    >>> p.start()
    >>> from time import sleep; sleep(10)
    >>> p.stop()
    Args:
        engine (Engine): Object containing info on the binary engine
        trigger_level (int): Number of chunk activations needed to trigger on_activation
                       Higher values add latency but reduce false positives
        threshold (float): From 0.0 to 1.0, the network output level required to consider a chunk "active"
        stream (BinaryIO): Binary audio stream to read 16000 Hz 1 channel int16
                           audio from. If not given, the microphone is used
        on_prediction (Callable): callback for every new prediction
        on_activation (Callable): callback for when the wake word is heard
    """

    def __init__(self, engine, trigger_level=3, threshold=0.5, stream=None,
                 on_prediction=lambda x: None, on_activation=lambda: None):
        self.engine = engine
        self.trigger_level = trigger_level
        self.stream = stream
        self.on_prediction = on_prediction
        self.on_activation = on_activation
        self.chunk_size = engine.chunk_size

        self.pa = None
        self.thread = None
        self.running = False
        self.is_paused = False
        self.detector = TriggerDetector(self.chunk_size, threshold, trigger_level)
        atexit.register(self.stop)

    def _wrap_stream_read(self, stream):
        """
        pyaudio.Stream.read takes samples as n, not bytes
        so read(n) should be read(n // sample_depth)
        """
        import pyaudio
        if getattr(stream.read, '__func__', None) is pyaudio.Stream.read:
            stream.read = lambda x: pyaudio.Stream.read(stream, x // 2, False)

    def start(self):
        """Start listening from stream"""
        if self.stream is None:
            from pyaudio import PyAudio, paInt16
            self.pa = PyAudio()
            self.stream = self.pa.open(
                16000, 1, paInt16, True, frames_per_buffer=self.chunk_size
            )

        self._wrap_stream_read(self.stream)

        self.engine.start()
        self.running = True
        self.is_paused = False
        self._handle_predictions()

    def stop(self):
        """Stop listening and close stream"""
        if self.thread:
            self.running = False
            if isinstance(self.stream, ReadWriteStream):
                self.stream.write(b'\0' * self.chunk_size)
            self.thread.join()
            self.thread = None

        self.engine.stop()

        if self.pa:
            self.pa.terminate()
            self.stream.stop_stream()
            self.stream = self.pa = None

    def pause(self):
        self.is_paused = True

    def play(self):
        self.is_paused = False

    def _handle_predictions(self):
        """Continuously check Precise process output"""
        while self.running:
            chunk = self.stream.read(self.chunk_size)

            if self.is_paused:
                continue

            prob = self.engine.get_prediction(chunk)
            self.on_prediction(prob)
            if self.detector.update(prob):
                self.on_activation()