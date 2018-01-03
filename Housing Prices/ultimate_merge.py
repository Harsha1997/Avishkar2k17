import csv
import pandas as pd
from collections import OrderedDict

with open("result.csv",'r') as fo:
	sec = pd.read_csv(fo)
	with open("missing.csv", "r") as f:
		first = pd.read_csv(f)	
		merged = first.merge(sec, on="House ID", how="outer")
		merged.to_csv("merged.csv", index=False)
		
	
		
	
