import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample
import re
import time


class Transcriber():
    def __init__(
            self,
            model_type="tiny.en",
            device="cpu",
            compute_type="int8",
            max_window_duration=1,
            start_prompt_delay=2
    ):
        """
        model_type: whisper model type (tiny, base, small, medium, large, turbo)
        device: use "cpu" or "cuda"
        compute_type: use "int8" or "float16"
        """
        self.whisper = WhisperModel(
            model_type, device=device, compute_type=compute_type)
        self.max_window_samples = int(max_window_duration * 16000)
        self.buffer = np.array([], dtype=np.float32)
        self.window_buffer = np.array([], dtype=np.float32)
        self.started_speaking = False
        self.stopped_speaking_time = None
        self.start_prompt_delay = start_prompt_delay
        self.sentences = []

    def transcribe_audio(self, chunk, old_transcript):
        if self.stopped_speaking_time is not None:
            current_time = time.time()
            time_diff = current_time - self.stopped_speaking_time
            if time_diff >= self.start_prompt_delay:
                self.sentences = []
                self.stopped_speaking_time = None
                return old_transcript, True

        sr, audio_data = chunk
        cleaned_audio = self.__clean_audio(sr, audio_data)
        window = self.__update_window_buffer(cleaned_audio)

        if window is None:
            return old_transcript, False

        self.__update_buffer(window)
        new_transcript = self.__transcribe(self.buffer)
        if (not self.sentences or new_transcript != self.sentences[-1]) and new_transcript:
            if not self.started_speaking:
                self.started_speaking = True
                self.stopped_speaking_time = None
                print("Started speaking", new_transcript)
                self.sentences.append(new_transcript)
            else:
                self.sentences[-1] = new_transcript
            return "\n".join(self.sentences), False
        else:
            self.reset()
            if self.started_speaking:
                self.stopped_speaking_time = time.time()
                self.started_speaking = False
                print("Stopped speaking")
            return old_transcript, False

    def reset(self):
        self.buffer = np.array([], dtype=np.float32)

    def __transcribe(self, audio_data) -> str:
        # add vad_filter
        segments, _ = self.whisper.transcribe(
            audio_data,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(onset=0.9, offset=0.5, min_speech_duration_ms=500, min_silence_duration_ms=1000)
        )
        segments = list(segments)
        return "".join([segment.text for segment in segments]).strip()

    def __clean_audio(self, sr, audio_data):
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        num_samples = round(len(audio_data) * float(16000) / sr)
        # Resample the audio data to the target sample rate
        resampled_audio = resample(audio_data, num_samples)
        audio_data = resampled_audio.astype(np.float32)
        # Normalize the audio (if necessary)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        return audio_data

    def __update_window_buffer(self, new_chunk):
        self.window_buffer = np.concatenate((self.window_buffer, new_chunk))
        if len(self.window_buffer) > self.max_window_samples:
            full_window = self.window_buffer[:self.max_window_samples]
            self.window_buffer = self.window_buffer[self.max_window_samples:]
            return full_window
        return None

    def __update_buffer(self, window):
        self.buffer = np.concatenate((self.buffer, window))


def find_index_ignore_special_chars(main_str, sub_str):
    # Remove special characters using regex (adjust the list of characters as needed)
    cleaned_main_str = re.sub(r'[^\w\s]', '', main_str)
    cleaned_sub_str = re.sub(r'[^\w\s]', '', sub_str)

    # Find index of the cleaned substring in the cleaned main string
    cleaned_index = cleaned_main_str.find(cleaned_sub_str)

    if cleaned_index != -1:
        # Map the cleaned index to the original string
        original_index = 0
        cleaned_counter = 0
        for i, char in enumerate(main_str):
            if re.match(r'\w|\s', char):  # Only count alphanumeric or whitespace characters
                if cleaned_counter == cleaned_index:
                    original_index = i
                    break
                cleaned_counter += 1

        return original_index
    return -1
