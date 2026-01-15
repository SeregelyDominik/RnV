from pathlib import Path

import librosa
import os

from rnv.converter import Converter
from rnv.ssl.models import WavLM
from rnv.ssl.models import mHuBERT147
from rnv.utils import get_vocoder_checkpoint_path

CHECKPOINTS_DIR = "checkpoints"
vocoder_checkpoint_path = get_vocoder_checkpoint_path(CHECKPOINTS_DIR)

# Initialize the converter with the vocoder checkpoint and rhythm conversion settings
# You can choose between "urhythmic" or "syllable" for rhythm_converter
# and "global" or "fine" for rhythm_model_type
converter = Converter(vocoder_checkpoint_path, rhythm_converter="syllable", rhythm_model_type="global") # or "fine" for fine-grained rhythm conversion

#feature_extractor = WavLM()
feature_extractor = mHuBERT147()
segmenter_path = Path("checkpoints/mhub_segmenter2.pth")
# Rhythm and Voice Conversion
target_style_feats_path = r"C:\testingstuff\mhuberttest\feats\LaciControl-mhubert147"
knnvc_topk = 4
lambda_rate = 1. #Path(r"C:\testingstuff\rythmmodeloutput\lacidiz\0_global_urhythmic_model.pth")
source_rhythm_model = Path(r"C:\testingstuff\mhuberttest\model\LaciDiz\0_syllable_models.pth") # ensure these correspond to the chosen rhythm model type
target_rhythm_model = Path(r"C:\testingstuff\mhuberttest\model\LaciControl\0_syllable_models.pth")

# Load wav and extract features
for file in os.listdir(r"C:\testingstuff\wavs\Lacidizart\wav"):
    source_wav_path = r"C:\testingstuff\wavs\Lacidizart\wav\\"+file
    source_wav, sr = librosa.load(source_wav_path, sr=None)
    source_feats = feature_extractor.extract_framewise_features(wav_path=source_wav_path, output_layer=None).cpu()
    wav = converter.convert(source_feats, target_style_feats_path, source_rhythm_model, target_rhythm_model,
                            segmenter_path, knnvc_topk, lambda_rate, source_wav=source_wav)
    output_path = r"testmhub\wav\\" + str(file)
    converter.convert(source_feats, target_style_feats_path, source_rhythm_model, target_rhythm_model, segmenter_path,
                      knnvc_topk, lambda_rate, source_wav=source_wav, save_path=output_path)



## or to write the output directly to a file


# Rhythm Conversion Only
#output_path = "output_rhythm_only.wav"
#converter.convert(source_feats, None, source_rhythm_model, target_rhythm_model, segmenter_path, source_wav=source_wav, save_path=output_path)

# Voice Conversion Only
#output_path = "output_voice_only.wav"
#converter.convert(source_feats, target_style_feats_path, None, None, segmenter_path, knnvc_topk, lambda_rate, save_path=output_path)