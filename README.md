# Wav2PWN Project Overview
Wav2PWN leverages the transferability of adversarial examples across self-supervised ASR models to infer the architecture of a target black-box model and launch targeted attacks. We first construct a behavioral response matrix that captures the attack success rates and transferability characteristics among multiple white-box SSL models (e.g., Wav2Vec2, HuBERT, Conformer). By comparing how a set of adversarial examples behave on the unknown black-box system against this reference matrix, we infer the most likely architecture of the target model. Once the closest white-box proxy model is identified, Wav2PWN generates targeted adversarial audio samples using it, achieving highly effective black-box attacks with minimal query requirements.

![Architecture](assets/arch.png)


# How to Run
## Download LibriSpeech Dataset
To run this project, you will need a set of clean original audio samples. We recommend using the LibriSpeech test-clean subset.

🔗 Download Link
Official site: http://www.openslr.org/12

Recommended subset: test-clean.tar.gz

After this step, your directory structure should look like:

```
Wav2PWN-Project/
├── LibriSpeech/
│ ├── test-clean/
│ └── ...
```

## 📥 Download Results

You can access our generated adversarial examples and transcription logs for eight test models from the link below:

🔗 [Download Result.zip from Google Drive](https://drive.google.com/file/d/your_file_id/view?usp=sharing)

 After downloading, unzip the file and place the <code>Result/</code> folder into the root of the project: 
```
Wav2PWN-Project/
├── Result/
│   ├── 84-121123-0005_perturbed.wav
│   ├── 84-121123-0005_pred.txt
│   └── transcription_log.csv
```

