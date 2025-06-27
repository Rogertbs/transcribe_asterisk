from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment

client = OpenAI()
speech_file_path = Path(__file__).parent / "speech.mp3"
speech_file_wav = Path(__file__).parent / "speech.wav"

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input="Olá, em que posso ajudar você hoje?",
    #instructions="Speak in a cheerful and positive tone.",
) as response:
    response.stream_to_file(speech_file_path)

    audio = AudioSegment.from_mp3(speech_file_path)
    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    audio.export(speech_file_wav, format="wav")
