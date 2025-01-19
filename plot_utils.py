import matplotlib.pyplot as plt
import sounddevice as sd


def plot_emission(emission):
    """
    Plots the emission probabilities as a heatmap.
    Parameters:
        emission (torch.Tensor): Emission matrix of shape [time_steps, labels].
    """
    fig, ax = plt.subplots()
    img = ax.imshow(emission.T, origin='lower', aspect='auto', interpolation='nearest')
    ax.set_title("Frame-wise class probability")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    fig.tight_layout()



def plot_trellis(trellis):
    fig, ax = plt.subplots()
    img = ax.imshow(trellis.T, origin="lower")
    ax.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
    ax.annotate("+ Inf", (trellis.size(0) - trellis.size(1) / 5, trellis.size(1) / 3))
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    fig.tight_layout()



def plot_trellis_with_path(trellis, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path.T, origin="lower")
    plt.title("The path found by backtracking")
    plt.tight_layout()


def plot_trellis_with_segments(trellis, segments, transcript, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start : seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    ax2.set_title("Label probability with and without repetation")
    xs, hs, ws = [], [], []
    for seg in segments:
        if seg.label != "|":
            xs.append((seg.end + seg.start) / 2 + 0.4)
            hs.append(seg.score)
            ws.append(seg.end - seg.start)
            ax2.annotate(seg.label, (seg.start + 0.8, -0.07))
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in path:
        label = transcript[p.token_index]
        if label != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)

    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.grid(True, axis="y")
    ax2.set_ylim(-0.1, 1.1)
    fig.tight_layout()


def plot_alignments(trellis, segments, word_segments, waveform, sample_rate):
    """
    Visualizes trellis with path and corresponding waveform segments.
    Parameters:
        trellis (torch.Tensor): Alignment trellis.
        segments (list): List of aligned segments.
        word_segments (list): List of word segment dictionaries with keys 'label', 'start', 'end', 'score'.
        waveform (torch.Tensor): Waveform tensor.
        sample_rate (int): Sampling rate of the audio.
    """
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start: seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1)

    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")
    ax1.set_facecolor("lightgray")
    ax1.set_xticks([])
    ax1.set_yticks([])

    for word in word_segments:
        x0, x1 = word['start'] - 0.5, word['end'] - 0.5
        ax1.axvspan(x0, x1, edgecolor="white", facecolor="none")

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    # The original waveform
    ratio = waveform.size(0) / sample_rate / trellis.size(0)
    ax2.specgram(waveform.numpy(), Fs=sample_rate)
    for word in word_segments:
        x0 = ratio * word['start']
        x1 = ratio * word['end']
        ax2.axvspan(x0, x1, facecolor="none", edgecolor="white", hatch="/")
        ax2.annotate(f"{word['score']:.2f}", (x0, sample_rate * 0.51), annotation_clip=False)

    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, sample_rate * 0.55), annotation_clip=False)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    fig.tight_layout()


def display_segment(i, waveform, trellis, word_segments, sample_rate):
    """
    Play a specific segment of the waveform based on alignment.

    Parameters:
        i (int): Index of the word segment to display.
        waveform (torch.Tensor): Audio waveform as a PyTorch tensor.
        trellis (torch.Tensor): Trellis used for alignment.
        word_segments (list): List of word segments with start, end, label, and score as attributes.
        sample_rate (int): Sampling rate of the audio.

    Returns:
        None
    """
    ratio = waveform.size(1) / trellis.size(0)
    word = word_segments[i]
    x0 = int(ratio * word.start)  # Access using dot notation
    x1 = int(ratio * word.end)    # Access using dot notation
    print(f"{word.label} ({word.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")

    # Extract the audio segment
    segment = waveform[:, x0:x1]

    # Downmix to mono if necessary
    if segment.size(0) > 1:
        segment = segment.mean(dim=0)

    # Play the audio segment
    sd.play(segment.numpy(), samplerate=sample_rate)
    sd.wait()  # Wait for playback to finish
