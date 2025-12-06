import os
import sys
import time
import logging
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- CONFIGURATION ---
MODEL_NAME = "facebook/wav2vec2-base-960h"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AudioAligner")


def load_audio_robust(audio_path, target_sample_rate=16000):
    """
    Loads audio using soundfile (most stable backend) and converts to
    Mono, 16kHz PyTorch Tensor.
    """
    t_start = time.time()

    if not os.path.exists(audio_path):
        logger.error(f"File not found at {audio_path}")
        sys.exit(1)

    logger.info(f"Loading audio file: {audio_path}...")
    try:
        # 1. Read directly with SoundFile
        data, sample_rate = sf.read(audio_path)
        t_read = time.time()
        logger.debug(f"File read in {t_read - t_start:.4f}s")
    except Exception as e:
        logger.error(f"Failed to read audio file. Error: {e}")
        sys.exit(1)

    # 2. Convert to Float Tensor
    waveform = torch.FloatTensor(data)

    # 3. Stereo to Mono
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=1)

    # 4. Resample if needed
    if sample_rate != target_sample_rate:
        logger.info(f"Resampling from {sample_rate}Hz to {target_sample_rate}Hz...")
        waveform = waveform.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform).squeeze()

    total_time = time.time() - t_start
    logger.info(f"Audio preprocessing finished in {total_time:.4f}s")
    return waveform


def compute_alignment_score(waveform, transcript):
    """
    Performs Forced Alignment and calculates a confidence score per word.
    """
    overall_start = time.time()

    # --- Step 1: Model Loading ---
    t0 = time.time()
    logger.info("Initializing Model (Wav2Vec2)...")

    # Suppress transformer warnings
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    t1 = time.time()
    logger.info(f"Model loaded in {t1 - t0:.4f}s")

    # --- Step 2: Forward Pass (Emissions) ---
    logger.info("Computing acoustic probabilities (Forward Pass)...")
    with torch.inference_mode():
        input_tensor = waveform.unsqueeze(0).to(DEVICE)
        logits = model(input_tensor).logits
        emissions = torch.log_softmax(logits, dim=-1)

    t2 = time.time()
    logger.info(f"Forward pass finished in {t2 - t1:.4f}s")

    # --- Step 3: Tokenization ---
    logger.info("Preparing text targets...")
    vocab = processor.tokenizer.get_vocab()
    clean_words = transcript.upper().split()
    target_tokens = []

    for word_idx, word in enumerate(clean_words):
        for char in word:
            if char in vocab:
                target_tokens.append((vocab[char], word_idx))
        if word_idx < len(clean_words) - 1:
            target_tokens.append((vocab['|'], -1))

    targets_flat = [t[0] for t in target_tokens]
    targets_tensor = torch.tensor([targets_flat], dtype=torch.int32, device=DEVICE)

    t3 = time.time()
    logger.debug(f"Tokenization finished in {t3 - t2:.4f}s")

    # --- Step 4: Forced Alignment ---
    logger.info("Running Forced Alignment Algorithm...")
    emission_lengths = torch.tensor([emissions.shape[1]], device=DEVICE)
    target_lengths = torch.tensor([targets_tensor.shape[1]], device=DEVICE)

    from torchaudio.functional import forced_align
    path, _ = forced_align(
        emissions,
        targets_tensor,
        emission_lengths,
        target_lengths,
        blank=0
    )

    t4 = time.time()
    logger.info(f"Alignment algorithm finished in {t4 - t3:.4f}s")

    # --- Step 5: Scoring Logic ---
    logger.info("Calculating word scores...")
    path = path[0].cpu().numpy()
    emissions_np = emissions[0].cpu().numpy()

    word_scores = {i: [] for i in range(len(clean_words))}
    target_ptr = 0

    for t in range(len(path)):
        token_id = path[t]
        if token_id == 0: continue

        score = emissions_np[t, token_id]

        if target_ptr < len(target_tokens):
            expected_token, word_idx = target_tokens[target_ptr]

            if word_idx != -1:
                word_scores[word_idx].append(score)

            if t + 1 < len(path) and path[t + 1] != token_id:
                target_ptr += 1

    results = []
    for i, word in enumerate(clean_words):
        scores = word_scores[i]
        if not scores:
            avg_score = -10.0
        else:
            avg_score = sum(scores) / len(scores)
        results.append({"word": word, "score": avg_score})

    t_end = time.time()
    logger.info(f"Scoring logic finished in {t_end - t4:.4f}s")
    logger.info(f"Total alignment process took: {t_end - overall_start:.4f}s")

    return results


if __name__ == "__main__":
    # --- USER SETTINGS ---
    AUDIO_FILE = "test_audio.wav"
    CORRECT_TEXT = "A DOG JUMPS OVER A FENCE AND BITES A WOMAN"

    logger.info(f"Target Text: '{CORRECT_TEXT}'")

    waveform = load_audio_robust(AUDIO_FILE)

    try:
        analysis = compute_alignment_score(waveform, CORRECT_TEXT)

        print("\n" + "=" * 60)
        print(f"{'WORD':<20} | {'SCORE':<20} | {'JUDGMENT'}")
        print("-" * 60)

        for item in analysis:
            score = item['score']
            if score > -2.0:
                status = "✅ Perfect"
            elif score > -4.5:
                status = "⚠️ Acceptable"
            else:
                status = "❌ INCORRECT / UNSURE"

            print(f"{item['word']:<20} | {score:.4f}               | {status}")

        print("=" * 60)

    except Exception as e:
        logger.critical(f"Script crashed: {e}", exc_info=True)