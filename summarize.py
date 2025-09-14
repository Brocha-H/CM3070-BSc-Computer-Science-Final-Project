#summarize.py

from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load small distilled BART model
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Force model to run on CPU
device = torch.device("cpu")
model.to(device)

def summarize_text(text, rate=0.3, max_input_length=1024):
    inputs = tokenizer(
        text,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt"
    )

    # Token count of the input
    input_length = len(inputs.input_ids[0])

    if input_length < 200:
        max_summary_length = max(10, int(input_length * rate * 2))
        min_summary_length = max(5, int(max_summary_length * 0.5))
    else:  
        max_summary_length = max(10, int(input_length * rate))
        min_summary_length = max(5, int(max_summary_length * 0.5))

    summary_ids = model.generate(
        inputs.input_ids.to(device),
        attention_mask=inputs.attention_mask.to(device),
        max_length=max_summary_length,
        min_length=min_summary_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_segment_files(transcribed_segments, rate):
    print("Starting summarization...")

    summarized = []
    for segment in transcribed_segments:
        speaker = segment["speaker"]
        transcript = segment["transcript"]
        word_count = len(transcript.strip().split())

        # If fewer than 10 words, skip summarization
        if word_count < 10:
            summary = transcript.strip()
        else:
            summary = summarize_text(transcript, rate)

        summarized.append({
            "speaker": speaker,
            "summary": summary
        })


    print("Summarization Complete!")

    return summarized