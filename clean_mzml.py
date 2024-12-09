import re
file_content = ""
with open("19CPTAC_LUAD_W_BI_20180730_KL_f08.mzML", 'r') as file:
    lines = file.readlines()

for i in range(len(lines)):
    lines[i] = (re.sub("(\"[^\"]*?[t|T]hermo[^\"]*?\")|(\"[^\"]*?[o|O]rbitrap[^\"]*?\")|(\"[^\"]*?[e|E]xactive[^\"]*?\")|(\"[^\"]*?[q|Q]uadrupole[^\"]*?\")", "\"\"", lines[i]))

with open("file_path", 'w') as file:
    for line in lines:
      file.write(line)