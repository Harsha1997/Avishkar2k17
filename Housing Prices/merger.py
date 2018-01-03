fout=open("merge_yes.csv","a")
f = open("Not_Known.csv","r+")
next(f)
for line in f:
         fout.write(line)
f.close() 
fout.close()
