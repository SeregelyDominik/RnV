@echo off

set VALUES=016 018 026 027 029 031 033 036 040 041 042

for %%V in (%VALUES%) do (
     python scripts/preprocess_speech_data.py 16000 D:\testingstuff\bme\HungarianDysartriaDatabase\lsa.tmit.bme.hu\samples\commands\%%V D:\testingstuff\bme\HungarianDysartriaDatabase\lsa.tmit.bme.hu\samples\commands\%%V_p
)

pause
