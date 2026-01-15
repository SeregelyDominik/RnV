@echo off

set VALUES=016 018 026 027 029 031 033 036 040 041 042 043 048

for %%V in (%VALUES%) do (
    python recipes/train_urhythmic_rhythm_model.py %%V global D:\testingstuff\bme\HungarianDysartriaDatabase\lsa.tmit.bme.hu\samples\commands\%%V_f-mhubert147 checkpoints/mhub_segmenter2.pth D:\testingstuff\bme\HungarianDysartriaDatabase\lsa.tmit.bme.hu\samples\models\%%V
)

pause
