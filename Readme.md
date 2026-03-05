Step 1 – Prepare your dataset
Place all your video or audio files inside the audios folder.

Example:
audios/video1.mp3
audios/video2.mp3
audios/video3.mp3

Step 2 – Convert audio into text (Transcription)
Run the Whisper script to generate subtitles from audio files.

This step converts speech → text with timestamps.

Output:
JSON files containing chunks of subtitles.

Step 3 – Chunk the transcript
Split the transcript into smaller segments (chunks).

Each chunk contains:

id

start time

end time

text

Output stored inside the json folder.

Example:
json/video1.json
json/video2.json

Step 4 – Generate embeddings
Convert each chunk of text into vector embeddings using the embedding model.

This step transforms text → numerical vectors for similarity search.

Embeddings stored inside a dataframe.

Step 5 – Save embeddings
Store the dataframe using Joblib.

This avoids recomputing embeddings every time.

Output:
embeddings.joblib

Step 6 – Ask questions
User enters a query.

The query is converted into an embedding.

Step 7 – Similarity search
Cosine similarity is used to compare:

User Question ↔ Stored Chunk Embeddings

Top matching chunks are retrieved.

Step 8 – Prompt creation
Relevant chunks are injected into a prompt.

The language model receives:

Context (retrieved chunks)

User question

Step 9 – Generate response
The LLM generates a grounded response using retrieved video content.

System Workflow Summary

videos → Audio Files → Whisper → Transcript → Chunking → JSON → Embeddings → Similarity Search → Prompt → LLM Response