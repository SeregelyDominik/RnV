import os
import sys

path = r"C:\testingstuff\mhuberttest\full_testing_environment\validation_hun2.txt"
with open(path, 'r', encoding='utf-8') as f:
    content = f.readlines()
    for line in content:
        lista = line.split(".")
        if lista[0].endswith("84"):
            print(line)