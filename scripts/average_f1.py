import sys

filename = sys.argv[1]
langs = ['en', 'de', 'fr', 'nl', 'he', 'pt', 'it', 'es']

results = {}
with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("language"):
            lan = line.split("=")[1]
        elif line.startswith("f1"):
            f1 = line.split(" = ")[1]
            results[lan] = float(f1)

print(results)

total = 0
for l in langs:
    total += results[l]
print(sum(results.values()) / len(results))
print(total/len(langs))
