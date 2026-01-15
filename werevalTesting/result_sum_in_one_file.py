import csv
import numpy as np
from collections import defaultdict
import argparse
import os

parser = argparse.ArgumentParser()
#parser.add_argument("speaker_folder_path", type=str)
parser.add_argument("--num_classes", type=int, required=True)
args = parser.parse_args()

engines = ["whisper", "beast", "mms"]

csv_names = (
    "rhythm_urhythmic_global.csv",
    "rhythm_urhythmic_fine.csv",
    "syllable_syllable_global.csv",
    "syllable_syllable_fine.csv",
    "knnvc-only_syllable_fine.csv",
    "knnvc_urhythmic_global.csv",
    "knnvc_urhythmic_fine.csv",
    "knnvc_syllable_global.csv",
    "knnvc_syllable_fine.csv",
    "space"
)
eredpath = r".\eredmenyek\\"
speakers = ["010", "016", "018", "026", "027", "029", "031", "033", "036", "040", "041", "042", "043", "048"]
with open("wer_counts.csv", "w", encoding="utf-8") as wer_out, \
     open("cer_counts.csv", "w", encoding="utf-8") as cer_out:
    for speaker in speakers:
        speaker_folder_path = eredpath + speaker
        wer_out.write(speaker_folder_path+", ")
        cer_out.write(speaker_folder_path+", ")


        for engine in engines:
            for csvname in csv_names:
                if csvname == "space":
                    wer_out.write(" ,")
                    cer_out.write(" ,")
                else:

                    path = os.path.join(speaker_folder_path, engine, csvname)
                    if not os.path.exists(path):
                        continue

                    # reset per-CSV stats
                    wer_per_class = defaultdict(list)
                    cer_per_class = defaultdict(list)
                    word_counts = defaultdict(list)
                    char_counts = defaultdict(list)

                    with open(path, newline='', encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            wer = float(row["wer"])
                            cer = float(row["cer"])
                            if wer < 0 or cer < 0:
                                continue

                            lab = int(row["class_label"])
                            wc = int(row["word_count"])
                            cc = int(row["char_count"])

                            wer_per_class[lab].append(wer)
                            cer_per_class[lab].append(cer)
                            word_counts[lab].append(wc)
                            char_counts[lab].append(cc)

                    wer_row = []
                    cer_row = []

                    for lab in range(args.num_classes):
                        if sum(word_counts[lab]) > 0:
                            wer_row.append(str(np.average(wer_per_class[lab], weights=word_counts[lab])))

                        if sum(char_counts[lab]) > 0:
                            cer_row.append(str(np.average(cer_per_class[lab], weights=char_counts[lab])))

                    wer_out.write("".join(wer_row) + ",")
                    cer_out.write("".join(cer_row) + ",")
        wer_out.write("\n")
        cer_out.write("\n")