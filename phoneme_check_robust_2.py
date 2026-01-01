import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import difflib
import eng_to_ipa as ipa
import warnings
import json

# 1. Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# --- CONFIGURATION ---
# Using the LV-60 model as per your snippet
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# Initialize Model (Global to avoid reloading)
print("Loading AI Model...")
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
except Exception as e:
    print(f"Error loading model. Ensure you have internet or cached models. Details: {e}")
    exit(1)


def get_phonemes_from_audio(audio_path):
    """
    Returns the raw IPA string from the audio using Wav2Vec2.
    """
    # Load and normalize audio
    data, sr = sf.read(audio_path)
    waveform = torch.FloatTensor(data)
    if waveform.ndim > 1: waveform = torch.mean(waveform, dim=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform.unsqueeze(0)).squeeze()

    # Inference
    with torch.no_grad():
        input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription[0]


def determine_word_color(word, expected_ipa, actual_ipa_chunk):
    """
    Compares expected vs actual IPA.
    Returns "red" if significant mismatch is found, otherwise "green".
    """
    # 1. Clean data
    clean_exp = expected_ipa.replace("ˈ", "").replace("ˌ", "")
    clean_act = actual_ipa_chunk

    # 2. Align Phonemes using SequenceMatcher
    matcher = difflib.SequenceMatcher(None, clean_exp, clean_act)

    # Heuristic: Map Phoneme Index -> Letter Index
    ratio = len(word) / max(1, len(clean_exp))

    # Initialize error mask
    letter_error_mask = [False] * len(word)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'delete':
            # Mismatch found. Map phoneme range to letter range.
            start_letter = int(i1 * ratio)
            end_letter = int(i2 * ratio)

            if start_letter == end_letter:
                end_letter += 1

            end_letter = min(end_letter, len(word))

            for k in range(start_letter, end_letter):
                if k < len(letter_error_mask):
                    letter_error_mask[k] = True

    # 3. Decision Logic
    # Count how many letters were flagged as incorrect
    error_count = sum(letter_error_mask)

    # Threshold Logic:
    # If any letters are wrong, we *could* mark it red.
    # But to be robust like Azure, let's say if > 30% of the word is wrong, mark red.
    # (For short words like "in", 1 wrong letter = 50% error -> Red).
    if len(word) > 0:
        error_percentage = error_count / len(word)
        if error_percentage > 0.3:  # 30% tolerance threshold
            return "red"

    return "green"


def analyze_pronunciation_json(target_text, audio_path):
    # 1. Get Phonemes from Audio
    try:
        heard_ipa_stream = get_phonemes_from_audio(audio_path)
    except Exception as e:
        return {"error": f"Failed to process audio: {str(e)}"}

    # Clean stream
    heard_ipa_stream = heard_ipa_stream.replace(" ", "")

    words = target_text.split()
    highlights = []
    stream_ptr = 0

    # 2. Iterate word by word
    for word in words:
        # Generate Expected IPA
        # Note: eng_to_ipa returns "word*" if it's unsure. We strip '*'
        word_ipa = ipa.convert(word).replace("*", "")
        clean_word_ipa = word_ipa.replace("ˈ", "").replace("ˌ", "").replace(" ", "")

        # Estimate length in the heard stream
        est_length = len(clean_word_ipa)

        # Grab window from stream
        # Look ahead a bit extra (+3 chars) to handle accents/elongation
        search_window = heard_ipa_stream[stream_ptr: stream_ptr + est_length + 3]

        # Determine Color
        color = determine_word_color(word, clean_word_ipa, search_window)

        # Add to results
        highlights.append({
            "word": word,
            "color": color
        })

        # Advance pointer (naive heuristic)
        stream_ptr += est_length

    # 3. Construct Final JSON Structure
    output = {
        "returned_sentence": target_text,
        "highlights": highlights
    }

    return output


if __name__ == "__main__":
    AUDIO_FILE = "test_audio.wav"
    TEXT = "COMMANDO JOHN LEAD THE TEAM TO VICTORY"

    # Run analysis
    result_json = analyze_pronunciation_json(TEXT, AUDIO_FILE)

    # Print formatted JSON
    print(json.dumps(result_json, indent=4))