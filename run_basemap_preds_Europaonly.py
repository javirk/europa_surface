import os

# quick script to run all basemaps freshly

# Europa geosize 500
command = 'python apply_LineaMapper_Europa_g500.py'
print(command)
os.system(command)

# Europa geosize 112
command = 'python apply_LineaMapper_Europa_g112.py'
print(command)
os.system(command)

