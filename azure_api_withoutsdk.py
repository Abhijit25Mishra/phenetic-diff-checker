import requests
import base64
import json
import logging
import os
import datetime

# --- CONFIGURATION ---
SPEECH_KEY =
SPEECH_REGION =


# --- LOGGER SETUP ---
def setup_logger(base_name="azure_rest_debug"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{base_name}_{timestamp}.log"
    logger = logging.getLogger("AzureRestDebug")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()

def get_audio_bytes(audio_source):
    """
    Helper to get raw audio bytes from either a URL or a local file path.
    """
    # CASE 1: It's a URL
    if audio_source.startswith("http://") or audio_source.startswith("https://"):
        logger.info(f"Downloading audio from URL: {audio_source}")
        try:
            response = requests.get(audio_source)
            response.raise_for_status()  # Raise error if 404/500
            logger.info(f"Download complete. Size: {len(response.content)} bytes")
            return response.content
        except Exception as e:
            logger.critical(f"Failed to download audio URL: {e}")
            return None


def check_pronunciation_rest(audio_source, reference_text):
    logger.info("=" * 50)
    logger.info("STARTING REST API PRONUNCIATION CHECK")
    logger.info("=" * 50)

    # 1. Prepare Endpoint
    url = f"https://{SPEECH_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-IN&format=detailed"

    # 2. Prepare Header
    pron_config = {
        "ReferenceText": reference_text,
        "GradingSystem": "HundredMark",
        "Granularity": "Phoneme",
        "Dimension": "Comprehensive",
        "EnableMiscue": True
    }

    pron_json = json.dumps(pron_config)
    pron_base64 = base64.b64encode(pron_json.encode('utf-8')).decode('utf-8')

    # 3. GET AUDIO DATA (From URL or Local)
    audio_data = get_audio_bytes(audio_source)
    if not audio_data:
        print("❌ Error: Could not load audio data.")
        return

    # 4. Headers
    headers = {
        "Ocp-Apim-Subscription-Key": SPEECH_KEY,
        "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        "Accept": "application/json",
        "Pronunciation-Assessment": pron_base64
    }

    # 5. Send Request
    logger.info("Sending POST request to Azure...")
    try:
        response = requests.post(url, headers=headers, data=audio_data)
        if response.status_code != 200:
            logger.error(f"HTTP Error: {response.status_code}")
            logger.error(f"Server Response: {response.text}")
            return
    except Exception as e:
        logger.critical(f"Connection Error: {e}")
        return

    # 6. Parse Response
    result = response.json()

    # --- PRINT COMPACT JSON (Single Line / Word Wrapped) ---
    print("\n" + "=" * 20 + " RAW JSON RESPONSE " + "=" * 20)
    print(json.dumps(result))
    print("=" * 60 + "\n")

    if result.get('RecognitionStatus') == 'Success':
        nbest = result.get('NBest', [])
        if not nbest:
            logger.warning("No NBest results found.")
            return

        best_attempt = nbest[0]

        def get_score(obj, key):
            if key in obj: return obj[key]
            if 'PronunciationAssessment' in obj:
                return obj['PronunciationAssessment'].get(key)
            return None

        acc_score = get_score(best_attempt, 'AccuracyScore')
        fluency_score = get_score(best_attempt, 'FluencyScore')
        display_text = result.get('DisplayText', '')

        print("--- RESULTS ---")
        print(f"Heard:    '{display_text}'")
        print(f"Accuracy: {acc_score}")
        print(f"Fluency:  {fluency_score}")

        print("\n--- WORD ANALYSIS ---")
        for word in best_attempt.get('Words', []):
            text = word.get('Word')
            accuracy = get_score(word, 'AccuracyScore')
            error_type = get_score(word, 'ErrorType')

            if accuracy is None: accuracy = 0

            icon = "✅"
            if accuracy < 80: icon = "⚠️"
            if accuracy < 60: icon = "❌"
            if error_type == "Omission": icon = "❌ MISSING"
            if error_type == "Insertion": icon = "⚠️ EXTRA"

            print(f"{icon} {text:<15} (Score: {accuracy})")

            if 'Phonemes' in word:
                print("   Phonemes: ", end="")
                for ph in word['Phonemes']:
                    ph_char = ph.get('Phoneme')
                    ph_acc = get_score(ph, 'AccuracyScore')
                    if ph_acc is None: ph_acc = 0

                    if ph_acc < 60:
                        print(f"[{ph_char} ❌] ", end="")
                    else:
                        print(f"/{ph_char}/ ", end="")
                print()

    elif result.get('RecognitionStatus') == 'NoMatch':
        logger.warning("No speech recognized.")
        print("No speech recognized.")
    else:
        logger.error(f"Recognition Failed. Status: {result.get('RecognitionStatus')}")


if __name__ == "__main__":
    # YOU CAN NOW PASS A URL HERE:
    # AUDIO_URL = "https://nkb-backend-ccbp-media-static.s3-ap-south-1.amazonaws.com/ccbp_beta/media/content_loading/uploads/f2293ad9-eb8d-455f-8ddd-c6590515833a_test_audio.wav"
    AUDIO_URL = "https://nkb-backend-ccbp-media-static.s3-ap-south-1.amazonaws.com/ccbp_beta/media/content_loading/uploads/07bb9f2d-e332-4469-b170-92284853f7e5_test_audio_wrong_rate%201.wav"
    TEXT = "The captain will lead the team to victory"

    check_pronunciation_rest(AUDIO_URL, TEXT)