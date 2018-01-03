import csv
with open("merge_yes.csv", "r") as f:
	first = {rows[0]: rows[1:] for rows in list(csv.reader(f))}

with open("house_prices.csv", "r") as f:
	for row in csv.reader(f):
		#print(row[1])
		if row[0] in first:
			first[row[0]].append(row[1])
#converting dictionary back to list

merged = [(k,) + tuple(v) for k, v in first.items()]

with open("train.csv", "w") as f:
    csv.writer(f).writerows(merged)
