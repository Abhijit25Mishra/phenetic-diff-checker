# import torch
# import torchaudio
# import soundfile as sf
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# import difflib
#
# # 1. Use the Phoneme-tuned model (Not the Word-tuned one)
# # This model speaks "IPA" instead of "English"
# MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
#
# print("Loading Phoneme Model...")
# processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
# model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
#
#
# def get_phonemes_from_audio(audio_path):
#     # Load audio (Standard 16k mono)
#     data, sr = sf.read(audio_path)
#     waveform = torch.FloatTensor(data)
#     if waveform.ndim > 1: waveform = torch.mean(waveform, dim=1)
#     if sr != 16000:
#         resampler = torchaudio.transforms.Resample(sr, 16000)
#         waveform = resampler(waveform.unsqueeze(0)).squeeze()
#
#     # Inference
#     with torch.no_grad():
#         input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
#         logits = model(input_values).logits
#         predicted_ids = torch.argmax(logits, dim=-1)
#
#         # Decode the IDs to Phoneme String
#         transcription = processor.batch_decode(predicted_ids)
#         return transcription[0]
#
#
# def compare_phonemes(expected_text, audio_path):
#     # 1. Get Expected Phonemes (Truth)
#     # We use a library to convert Text -> IPA
#     import eng_to_ipa as ipa
#     expected_ipa = ipa.convert(expected_text)
#
#     # 2. Get Actual Phonemes (AI Listener)
#     print("Listening...")
#     actual_ipa = get_phonemes_from_audio(audio_path)
#
#     print(f"\nTarget Text:   {expected_text}")
#     print(f"Expected IPA:  /{expected_ipa}/")
#     print(f"Heard IPA:     /{actual_ipa}/")
#
#     # 3. Highlight Differences
#     # We clean up the strings to make comparison fair
#     seq1 = list(expected_ipa.replace("ˈ", "").replace("ˌ", ""))  # Remove stress markers
#     seq2 = list(actual_ipa)
#
#     matcher = difflib.SequenceMatcher(None, seq1, seq2)
#
#     print("\n--- DIFF ANALYSIS ---")
#     print(f"{'EXPECTED':<10} | {'HEARD':<10} | {'STATUS'}")
#     print("-" * 35)
#
#     for tag, i1, i2, j1, j2 in matcher.get_opcodes():
#         seg_exp = "".join(seq1[i1:i2])
#         seg_act = "".join(seq2[j1:j2])
#
#         if tag == 'equal':
#             print(f"{seg_exp:<10} | {seg_act:<10} | ✅ Match")
#         elif tag == 'replace':
#             print(f"{seg_exp:<10} | {seg_act:<10} | ❌ Wrong Sound")
#         elif tag == 'insert':
#             print(f"{'-':<10} | {seg_act:<10} | ⚠️ Extra Sound")
#         elif tag == 'delete':
#             print(f"{seg_exp:<10} | {'-':<10} | ❌ Missing")
#
#
# if __name__ == "__main__":
#     AUDIO_FILE = "test_audio.wav"
#     TEXT = "COMMANDO JOHN LEAD THE TEAM TO VICTORY"
#
#     compare_phonemes(TEXT, AUDIO_FILE)


import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import difflib
import eng_to_ipa as ipa
import warnings
import re

# 1. Suppress the annoying Regex warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# --- CONFIGURATION ---
# We use the XLSR model (Better for Accents) as discussed
# MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


print("Loading AI Model...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)


# --- HELPER: ANSI COLORS ---
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def get_phonemes_from_audio(audio_path):
    """
    Returns the raw IPA string from the audio.
    """
    data, sr = sf.read(audio_path)
    waveform = torch.FloatTensor(data)
    if waveform.ndim > 1: waveform = torch.mean(waveform, dim=1)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform.unsqueeze(0)).squeeze()

    with torch.no_grad():
        input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription[0]


def highlight_errors_in_word(word, expected_ipa, actual_ipa_chunk):
    """
    Compares the expected sounds vs heard sounds for a SINGLE word,
    and returns the word string with mispronounced letters colored RED.
    """
    # 1. Clean data
    # Remove stress markers for easier comparison
    clean_exp = expected_ipa.replace("ˈ", "").replace("ˌ", "")
    clean_act = actual_ipa_chunk

    # 2. Align Phonemes
    matcher = difflib.SequenceMatcher(None, clean_exp, clean_act)

    # We need to map Phoneme Index -> Letter Index
    # Heuristic: Simple Ratio Mapping
    # If word has 5 letters and 4 sounds, sound[0] maps to letters[0..1.2]
    ratio = len(word) / max(1, len(clean_exp))

    # We build a boolean mask: True = Error, False = Correct
    # Initialize all letters as Correct (False)
    letter_error_mask = [False] * len(word)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'delete':
            # Mismatch found in Expected Phonemes range [i1:i2]
            # Map this range to Letter indices
            start_letter = int(i1 * ratio)
            end_letter = int(i2 * ratio)

            # Ensure we highlight at least one letter if the range is small
            if start_letter == end_letter:
                end_letter += 1

            # Clamp to word length
            end_letter = min(end_letter, len(word))

            # Mark these letters as errors
            for k in range(start_letter, end_letter):
                if k < len(letter_error_mask):
                    letter_error_mask[k] = True

    # 3. Construct the Colored String
    final_str = ""
    for i, char in enumerate(word):
        if letter_error_mask[i]:
            # RED + UNDERLINE + BOLD
            final_str += f"{Color.RED}{Color.UNDERLINE}{Color.BOLD}{char}{Color.END}"
        else:
            final_str += f"{Color.GREEN}{char}{Color.END}"

    return final_str


def analyze_pronunciation(target_text, audio_path):
    print("Listening to audio...")
    heard_ipa_stream = get_phonemes_from_audio(audio_path)

    # Clean up the stream (remove extra spaces)
    heard_ipa_stream = heard_ipa_stream.replace(" ", "")

    print("\n" + "=" * 50)
    print("      PRONUNCIATION REPORT")
    print("=" * 50 + "\n")

    words = target_text.split()
    results = []

    # Pointer to track where we are in the heard stream
    stream_ptr = 0

    print(f"Original: {target_text}")
    print("Feedback: ", end="")

    for word in words:
        # 1. Get Expected IPA for this specific word
        word_ipa = ipa.convert(word)
        # remove stress markers for length calculation
        clean_word_ipa = word_ipa.replace("ˈ", "").replace("ˌ", "").replace(" ", "")

        # 2. Extract the corresponding chunk from the Heard Stream
        # Heuristic: We assume the user spoke at roughly the same speed.
        # We grab a chunk of phonemes from the stream roughly equal to the expected length
        # (plus a little buffer to be safe)
        est_length = len(clean_word_ipa)

        # Dynamic Window: Grab likely corresponding sounds
        # We look ahead in the stream to find the best match for this word
        search_window = heard_ipa_stream[stream_ptr: stream_ptr + est_length + 3]

        # 3. Highlight Letters
        colored_word = highlight_errors_in_word(word, clean_word_ipa, search_window)

        print(colored_word + " ", end="")

        # Advance pointer
        # We advance by however much of the window "matched" best?
        # For prototype, we just advance by expected length to keep it simple.
        stream_ptr += len(clean_word_ipa)

    print("\n\n" + "=" * 50)
    print(f"Legend: {Color.GREEN}Correct{Color.END} | {Color.RED}{Color.UNDERLINE}Incorrect{Color.END}")


if __name__ == "__main__":
    AUDIO_FILE = "test_audio.wav"
    TEXT = "COMMANDO JOHN LEAD THE TEAM TO VICTORY"

    analyze_pronunciation(TEXT, AUDIO_FILE)