import urllib.request

url = "https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/model/sam3_base_predictor.py"
response = urllib.request.urlopen(url)
content = response.read().decode('utf-8')

with open("c:/Users/ANSH/.gemini/antigravity/scratch/ChronoGuard/sam3_base_predictor_full.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Downloaded successfully! Length:", len(content))
