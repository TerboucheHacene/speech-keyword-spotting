import requests

url = "http://0.0.0.0/api/predict"
audio_file_path = (
    "artifacts/SpeechCommands/speech_commands_v0.02/bed/0a7c2a8d_nohash_0.wav"
)

# Read audio file in binary mode
files = [
    ("waveform", (audio_file_path, open(audio_file_path, "rb"), "audio/wav")),
]
payload = {"sr": 16000, "duration": 1.0, "offset": 0.0}
headers = {}

response = requests.request(
    "POST",
    url,
    headers=headers,
    params=payload,
    files=files,
)

# results = [response.json()]
print(response.json())
