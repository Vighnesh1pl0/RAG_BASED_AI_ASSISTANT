import whisper
import json
import os

model = whisper.load_model("base")

audio_folder = "audios"

# create json folder
os.makedirs("json", exist_ok=True)

for file in os.listdir(audio_folder):

    if file.endswith(".mp3"):

        print("Processing:", file)

        result = model.transcribe(f"{audio_folder}/{file}")

        chunks = []

        for segment in result["segments"]:

            chunks.append({
                "id": len(chunks),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        with open(f"json/{file}.json", "w") as f:
            json.dump(chunks, f, indent=4)

        print("Saved JSON\n")
