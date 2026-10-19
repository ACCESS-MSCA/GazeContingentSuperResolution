"""The gaze model's fixed input and output, set by its training: the one place they are defined.

Kept free of torch, so the stream side (gaze.tracker, stream.control, the control page server)
can import it without loading the model.
"""

MODEL_HZ = 30  # samples per second the model takes (frames + head pose) and predicts at
PAST = 15      # samples in its input window (500 ms)
FUTURE = 20    # steps it predicts, one per sample: step n is n / MODEL_HZ s ahead (20 = 666 ms)
