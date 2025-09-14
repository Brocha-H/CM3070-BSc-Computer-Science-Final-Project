#app.py

from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from collections import defaultdict
from pydub import AudioSegment
import os
import random
from diarize import diarize_audio
from transcribe import transcribe_segments
from summarize import summarize_segment_files

app = Flask(__name__)

# Define folder paths
UPLOAD_FOLDER = 'uploads'
SEGMENTS_FOLDER = 'segments'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTS_FOLDER'] = SEGMENTS_FOLDER


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith(".wav"):
                os.remove(file_path) 



@app.route('/')
def landing():
    return render_template("landing.html")


@app.route('/upload', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':


        clear_folder(UPLOAD_FOLDER)
        clear_folder(SEGMENTS_FOLDER)

        if 'audiofile' not in request.files:
            return redirect(request.url)
        file = request.files['audiofile']
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        rate_input = request.form.get('rate')

        try:
            rate = float(rate_input)
        except (TypeError, ValueError):
            rate = 0.65

        num_speakers_input = request.form.get('num_speakers')
        print("Form input for num_speakers:", num_speakers_input)  # debug

        try:
            num_speakers = int(num_speakers_input)

        except (TypeError, ValueError):
            num_speakers = 2

        print("Using num_speakers =", num_speakers)


        # Step 1: Diarize and save segments
        diarized_segments = diarize_audio(filepath, SEGMENTS_FOLDER, num_speakers=num_speakers)

        # Step 2: Transcribe segments
        transcribed_segments = transcribe_segments(diarized_segments)


        # Step 3: Summarize transcripts
        summaries = summarize_segment_files(transcribed_segments, rate)

        return render_template("result.html", transcript=summaries)

    return render_template("upload.html")



if __name__ == "__main__":
    app.run(debug=True)