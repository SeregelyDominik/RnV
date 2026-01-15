@echo off

set VALUES=016 018 026 027 029 031 033 036 040 041 042

for %%V in (%VALUES%) do (
      python scripts/extract_dataset_embeddings.py mhubert147 D:\testingstuff\bme\HungarianDysartriaDatabase\lsa.tmit.bme.hu\samples\commands\%%V_p D:\testingstuff\bme\HungarianDysartriaDatabase\lsa.tmit.bme.hu\samples\commands\%%V_f
)

pause
