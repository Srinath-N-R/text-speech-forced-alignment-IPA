import torch
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor


def collapse_tokens(tokens):
    prev_token = None
    out = []
    for token in tokens:
        if token != prev_token and prev_token is not None:
            out.append(prev_token)
        prev_token = token
    return out


def clean_token_ids(token_ids, PAD_ID, EMPTY_ID):
    token_ids = [x for x in token_ids if x not in [PAD_ID, EMPTY_ID]]
    token_ids = collapse_tokens(token_ids)
    return token_ids


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens)).to(emission.device)
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.logsumexp(
            torch.stack([
                trellis[t, 1:] + emission[t, blank_id],        # Stay
                trellis[t, :-1] + emission[t, tokens[1:]]     # Transition
            ]), dim=0
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        assert t > 0

        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Handle the initial blank token
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def process_seg(seg, ratio, last_non_empty):
    start_frame = (seg.start * ratio)
    end_frame = (seg.end * ratio)
    if end_frame > last_non_empty:
        end_frame = last_non_empty 
    duration_frames = end_frame - start_frame
    return {"start_sec": start_frame, "end_sec": end_frame, "duration_sec": duration_frames}


def constrained_viterbi_alignment(emission, tokens, blank_id=0):
    """
    Perform a constrained Viterbi alignment that strictly follows the given token sequence.
    """
    num_frames, num_tokens = emission.size(0), len(tokens)
    trellis = torch.full((num_frames, num_tokens), -float('inf'))
    trellis[:, 0] = torch.cumsum(emission[:, blank_id], dim=0)

    for t in range(1, num_frames):
        for j in range(1, num_tokens):
            trellis[t, j] = max(
                trellis[t - 1, j] + emission[t, blank_id],
                trellis[t - 1, j - 1] + emission[t, tokens[j]]
            )
    
    # Backtracking to find the best path
    path = backtrack(trellis, emission, tokens, blank_id=blank_id)
    return path


class NewAligner:
    def __init__(self, sampling_rate=16000):
        self._device = torch.device("cpu")
        self.sampling_rate = sampling_rate

        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )

        # Load model in float32 to avoid potential half-precision issues
        self.model = Wav2Vec2ForCTC.from_pretrained("srinathnr/ipa-eng-asr-wav2vec2", torch_dtype=torch.float32).to(self._device)

        self.tokenizer = AutoTokenizer.from_pretrained("srinathnr/ipa-eng-asr")

        self.PAD_ID = self.tokenizer .encode("[PAD]")[0]   # e.g. [PAD]
        self.EMPTY_ID = self.tokenizer .encode(" ")[0]     # e.g. " " (space)

        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=self.tokenizer )


    def align_batch(self, mod_waveforms, phonemes):
        """
        waveforms: Can be a list of 1D numpy arrays or a list of 1D torch.Tensors
                representing multiple audio clips.

        Return: A list of (aln_hard, aln_mask) pairs, one per example in the batch.

        Note: Each example can have different lengths -> different # of tokens,
            so we store alignment results in a list rather than a single tensor.
        """
        # 1) Use the processor to batch-encode
        inputs = self.processor(
            mod_waveforms,
            sampling_rate=self.sampling_rate,
            return_tensors='pt',
            padding=True
        )
        input_values = inputs["input_values"].squeeze(0).to(self._device)

        with torch.no_grad():
            emissions = self.model(input_values).logits
        
        batch_size = len(emissions)
        segments = []
        for b in range(batch_size):
            emission_b = emissions[b]
            phonemes_b = phonemes[b]
            new_pred_ids_b = [self.tokenizer.convert_tokens_to_ids(p) for p in phonemes_b]
            tokens_b = clean_token_ids(new_pred_ids_b, self.PAD_ID, self.EMPTY_ID)
            tokens_b = torch.tensor(tokens_b).to(self._device)
            trellis_b = get_trellis(emission_b, tokens_b)
            path_b = constrained_viterbi_alignment(emission_b, tokens_b)

            aln_mask = torch.zeros_like(trellis_b, dtype=torch.int)
            for p in path_b:
                aln_mask[p.time_index, p.token_index] = 1

            segments_b = merge_repeats(path_b, phonemes_b)
            ratio = mod_waveforms.size(1) / self.sampling_rate / (trellis_b.size(0))
            mod_waveform_b = mod_waveforms[b]
            last_non_empty = mod_waveform_b.nonzero(as_tuple=False).max().item() / self.sampling_rate
            durations = {i: (seg.label, process_seg(seg, ratio, last_non_empty)) for i, seg in enumerate(segments_b)}
            
            segments.append(durations)
        return segments

