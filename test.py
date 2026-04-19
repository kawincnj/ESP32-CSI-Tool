import re
line = "winkawin [12, 23, 23, 3]"
ans = re.search(r"\[(.*?)\]", line)
csi_string = ans.group(1)
csi_data = [int(x) for x in csi_string.split(',')]
print(ans)