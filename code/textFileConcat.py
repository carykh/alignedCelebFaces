filenames = []
for i in range(0,13000,100):
    filenames.append("names/name"+str(i)+".txt")
with open('names/allNames.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
