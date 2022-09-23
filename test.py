import pickle

filename = "./environment/layouts/originalClassic.lay"
f = open(filename)
for line in f:
    x = 0
map = [line.strip() for line in f]
