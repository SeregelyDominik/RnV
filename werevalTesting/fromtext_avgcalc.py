import csv
import numpy as np
from collections import defaultdict
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("speaker_folder_path", type=str)
parser.add_argument("--num_classes", type=int, required=True)
args = parser.parse_args()


folderek = ["whisper", "beast", "mms"]
csvk = ["rhythm_urhythmic_global.csv", "rhythm_urhythmic_fine.csv", "syllable_syllable_global.csv",
        "syllable_syllable_fine.csv", "knnvc-only_syllable_fine.csv", "knnvc_urhythmic_global.csv",
        "knnvc_urhythmic_fine.csv", "knnvc_syllable_global.csv", "knnvc_syllable_fine.csv", "space"]


for folder in folderek:
    for c in csvk:
        if c == "space":
            with open('wer_counts.csv', 'a', newline='', encoding="utf-8") as g:
                g.write(" ,")

            with open('cer_counts.csv', 'a', newline='', encoding="utf-8") as k:
                k.write(" ,")
        else:

            with open(os.path.join(args.speaker_folder_path, folder, c), newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)

                wer_errors = defaultdict(float)
                wer_words = defaultdict(int)
                cer_errors = defaultdict(float)
                cer_chars = defaultdict(int)

                for row in reader:
                    wer = float(row["wer"])
                    cer = float(row["cer"])
                    if wer < 0 or cer < 0:
                        continue

                    lab = int(row["class_label"])
                    wc = int(row["word_count"])
                    cc = int(row["char_count"])

                    wer_errors[lab] += wer * wc
                    wer_words[lab] += wc

                    cer_errors[lab] += cer * cc
                    cer_chars[lab] += cc


            with open('wer_counts.csv', 'a', newline='', encoding="utf-8") as g:
                for lab in range(args.num_classes):
                    if wer_words[lab] > 0:
                        wer = wer_errors[lab] / wer_words[lab]
                        g.write(f"{wer},")
                    else:
                        wer = -1


            with open('cer_counts.csv', 'a', newline='', encoding="utf-8") as k:
                for lab in range(args.num_classes):
                    if cer_chars[lab] > 0:
                        cer = cer_errors[lab] / cer_chars[lab]
                        k.write(f"{cer},")
                    else:
                        cer = -1


