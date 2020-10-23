import sys
import pandas as pd
import os

filename = sys.argv[1]
seeds = [1, 2, 3, 4, 5]
langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'tr', 'ur', 'vi', 'zh']

seeded_results = []
for seed in seeds:
  results = {}
  c_f = filename + "_s{}/test_results.txt".format(seed)
  if os.path.exists(c_f):
      with open(c_f, 'r') as f:
          for line in f:
              line = line.strip()
              if line.startswith("language"):
                  lan = line.split("=")[1]
              elif line.startswith("f1"):
                  f1 = line.split(" = ")[1]
                  results[lan] = float(f1)*100
  
  #print(results)
  #print(sum(results.values()) / len(results))
  seeded_results.append(results)

df = pd.DataFrame(seeded_results)
print(df)
print(df.mean(axis=1))
print("ave over random seeds:", df.mean(axis=1).mean())

print(df.mean(axis=1))
print(pd.DataFrame(df, columns=langs).mean(axis=0).mean())
df.to_csv(filename+".csv")
