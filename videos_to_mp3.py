import os
import subprocess

files = os.listdir("videos")

for file in files:

    if file.endswith(".mp4"):

        input_file = f"videos/{file}"
        output_file = f"audios/{file[:-4]}.mp3"

        print("Converting:", file)

        subprocess.run(["ffmpeg", "-i", input_file, output_file])

        print("Done\n")




