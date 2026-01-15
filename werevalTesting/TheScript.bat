@echo off
setlocal EnableDelayedExpansion

set class[0]=urhythmic
set type[0]=global
set mode[0]=rhythm

set class[1]=urhythmic
set type[1]=fine
set mode[1]=rhythm

set class[2]=syllable
set type[2]=global
set mode[2]=syllable

set class[3]=syllable
set type[3]=fine
set mode[3]=syllable

set class[4]=syllable
set type[4]=fine
set mode[4]=knnvc-only

set class[5]=urhythmic
set type[5]=global
set mode[5]=knnvc

set class[6]=urhythmic
set type[6]=fine
set mode[6]=knnvc

set class[7]=syllable
set type[7]=global
set mode[7]=knnvc

set class[8]=syllable
set type[8]=fine
set mode[8]=knnvc

set size=9

set VALUES=043 048

for %%V in (%VALUES%) do (

    python "C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\txt_mover.py" %%V

    for /L %%i in (0,1,%size%-1) do (

    set CLASS=!class[%%i]!
    set TYPE=!type[%%i]!
    set MODE=!mode[%%i]!


    python "C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\convert.py" %%V !CLASS! !TYPE! "D:\testingstuff\bme\HungarianDysartriaDatabase\lsa.tmit.bme.hu\samples\commands\%%V_f-mhubert147" "C:\testingstuff\mhuberttest\feats\Szindbad-mhubert147" "C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\checkpoints" "C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\checkpoints\mhub_segmenter2.pth" !MODE! "C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\testmhub\wavs" "D:\testingstuff\bme\HungarianDysartriaDatabase\lsa.tmit.bme.hu\samples\commands\%%V_p"

    mkdir "eredmenyek\%%V\beast" 2>nul
    mkdir "eredmenyek\%%V\mms" 2>nul
    mkdir "eredmenyek\%%V\whisper" 2>nul


    py -3.10 .\eval_with_beast2.py HunDys "C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\testmhub\wavs" "eredmenyek\%%V\beast\!MODE!_!CLASS!_!TYPE!.csv" C_%%V

    py -3.10 .\eval_with_MMS.py HunDys "C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\testmhub\wavs" "eredmenyek\%%V\mms\!MODE!_!CLASS!_!TYPE!.csv"

    python .\eval_with_whisper.py HunDys "C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\testmhub\wavs" "eredmenyek\%%V\whisper\!MODE!_!CLASS!_!TYPE!.csv"

    )

)

endlocal