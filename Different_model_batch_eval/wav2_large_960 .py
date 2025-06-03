import os
import random
import csv
import torch
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from difflib import SequenceMatcher
import re

# Configurations
ROOT_DIR = "/home/mod/Documents/Audio-attack_statistic/fackbook_wav2_target_attack/LibriSpeech/dev-clean/"
OUTPUT_DIR = "/home/mod/Documents/Audio-attack_statistic/Other_wav2_model_test_100_samples/large_960/large_960h_OPEN THE EMAIL ATTACHMENT AND INSTALL THE INCLUDED SOFTWARE/"

TARGET_TEXT = "TURN OFF SECURITY CAMERA"



#DELETE FILES
#SHUT DOWN
#DOWNLOAD THE CONFIDENTIAL REPORT NOW
#TURN OFF SECURITY CAMERA
#OPEN THE EMAIL ATTACHMENT AND INSTALL THE INCLUDED SOFTWARE




files_number = 2
EPSILON = 0.08
lr = 0.001
total_steps = 2000
rw = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_FILE = os.path.join(OUTPUT_DIR, "attack_results.csv")

# Initialize counters
Mis_Trans = 0

# Load model and processor
model_name = "Facebook/wav2vec2-large-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(DEVICE)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Function to split target text into characters with | for spaces
def split_target_text(text):
    return " ".join(list(text.replace(" ", "|").upper()))

def split_original_text(text):
    return " ".join(list(text.replace(" ", "|").upper()))

# Load audio
def load_audio(file_path, sampling_rate=16000):
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    return audio, sr

# Calculate weighted accuracy
def calculate_weighted_accuracy(predicted, target, tolerance_factor=1):
    predicted_clean = predicted.replace(" ", "").replace("|", "")
    target_clean = target.replace(" ", "").replace("|", "")
    matcher = SequenceMatcher(None, predicted_clean, target_clean)
    similarity = matcher.ratio()
    max_length = max(len(predicted_clean), len(target_clean))
    accuracy = (similarity * (len(target_clean) + (tolerance_factor ** 0.5)) / max_length) * 100
    return min(accuracy, 100)

# Save spectrogram
def save_spectrogram(audio, sampling_rate, title, file_name):
    plt.figure(figsize=(10, 6))
    spectrogram = librosa.stft(audio)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    librosa.display.specshow(spectrogram_db, sr=sampling_rate, x_axis="time", y_axis="hz", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# Write results to CSV
def write_to_csv(file_name, original, adversarial, success_rate, mis_trans_rate):
    with open(CSV_FILE, mode='a', newline='', buffering=1) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([file_name, original, adversarial, success_rate, mis_trans_rate])
        csvfile.flush()

# Check already processed files
def get_processed_files():
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header
            return {row[0] for row in reader}
    return set()

# Main processing
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Prepare CSV file
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', buffering=1) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Audio File", "Original Transcription", "Adversarial Transcription", "Success_rate (%)", "Mis-Trans_rate (%)"])
        csvfile.flush()

processed_files = get_processed_files()

random.seed(1234)
for root, dirs, files in os.walk(ROOT_DIR):
    flac_files = [f for f in files if f.endswith('.flac')]

    if len(flac_files) < 2:
        selected_files = flac_files
    else:
        selected_files = random.sample(flac_files, files_number)

    for flac_file in selected_files:
        if flac_file in processed_files:
            print(f"Skipping already processed file: {flac_file}")
            continue

        input_audio_path = os.path.join(root, flac_file)
        output_folder = os.path.join(OUTPUT_DIR, flac_file.replace(".flac", ""))
        os.makedirs(output_folder, exist_ok=True)

        # Load and process audio
        speech_array, sampling_rate = load_audio(input_audio_path)
        input_values = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values.to(DEVICE)

        #split_text = split_target_text(TARGET_TEXT)
        #target_ids = processor.tokenizer(split_text, return_tensors="pt").input_ids.to(DEVICE)
        target_ids = processor.tokenizer(TARGET_TEXT, return_tensors="pt").input_ids.to(DEVICE)
        print("===============================")
        print(target_ids)

        # Initialize perturbation
        perturbation = torch.zeros_like(input_values, requires_grad=True, device=DEVICE)
        optimizer = torch.optim.Adam([perturbation], lr=lr)

        match_count = 0
        match_threshold = 5

        # Adversarial attack optimization
        for step in tqdm(range(total_steps)):
            perturbed_input = input_values + perturbation
            perturbed_input = torch.clamp(perturbed_input, -1.0, 1.0)

            outputs = model(perturbed_input)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1).permute(1, 0, 2)

            input_lengths = torch.tensor([log_probs.size(0)] * log_probs.size(1), dtype=torch.long, device=DEVICE)
            target_lengths = torch.full((target_ids.size(0),), target_ids.size(1), dtype=torch.long, device=DEVICE)

            loss_ctc = torch.nn.functional.ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
            perturbation_regularization = torch.norm(perturbation, p=2)
            loss = loss_ctc + rw * perturbation_regularization

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([perturbation], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                perturbation.data = torch.clamp(perturbation.data, -EPSILON, EPSILON)

            if step % 100 == 0:
                adversarial_transcription = processor.decode(torch.argmax(outputs.logits, dim=-1)[0], skip_special_tokens=True)
                print(f"Step {step}, Loss: {loss.item()}, Perturbed Transcription: {adversarial_transcription}")

            # if adversarial_transcription.strip() == re.sub(r'\s+', ' ', split_text.replace("|", " ")).strip():
            #     match_count += 1
            #     if match_count >= match_threshold:
            #         print("Target text matched, stopping optimization.")
            #         break
            # else:
            #     match_count = 0
            if adversarial_transcription.strip().upper() == TARGET_TEXT.strip().upper():
                match_count += 1
                if match_count >= match_threshold:
                    print("Target text matched, stopping optimization.")
                    break
            else:
                match_count = 0

        # Save adversarial audio
        with torch.no_grad():
            perturbed_input = input_values + perturbation
            perturbed_input = torch.clamp(perturbed_input, -1.0, 1.0)
            perturbed_audio = perturbed_input.cpu().detach().numpy().squeeze()
            output_audio_path = os.path.join(output_folder, flac_file.replace(".flac", "_perturbed.wav"))
            sf.write(output_audio_path, perturbed_audio, sampling_rate)

        # Get transcriptions
        original_transcription = processor.decode(torch.argmax(model(input_values).logits, dim=-1)[0])
        success_rate = calculate_weighted_accuracy(adversarial_transcription, TARGET_TEXT, tolerance_factor=1)

        split_original = split_original_text(original_transcription)

        # Determine Mis_Trans (untargeted attack success)
        if adversarial_transcription != original_transcription:
            Mis_Trans = 100  # Attack successful
        else:
            Mis_Trans = 0  # Attack failed

        # check perturbed_audio if include NaN or inf
        if not np.isfinite(perturbed_audio).all():
            print(f"Warning: perturbed_audio contains NaN or Inf values for {flac_file}. Skipping spectrogram generation.")
            write_to_csv(flac_file, "ERROR", "ERROR", "N/A", "N/A")
            continue 
        else:
            # Save spectrograms
            save_spectrogram(speech_array, sampling_rate, "Original Audio Spectrogram", os.path.join(output_folder, "original_spectrogram.png"))
            save_spectrogram(perturbed_audio, sampling_rate, "Perturbed Audio Spectrogram", os.path.join(output_folder, "perturbed_spectrogram.png"))

        # Write to CSV
        write_to_csv(flac_file, original_transcription, adversarial_transcription, round(success_rate, 2), round(Mis_Trans, 2))

        print(f"Processed: {flac_file}")
        print(f"Original: {original_transcription}")
        print(f"Adversarial: {adversarial_transcription}")
        print(f"Success_rate: {success_rate:.2f}%")
        print(f"Mis-Trans_rate: {Mis_Trans:.2f}%")
        print("==========================================")

print("Batch processing complete. Results saved in CSV.")