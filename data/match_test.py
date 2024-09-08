import numpy as np
# read test dataset logits tensor from csv
breed = []
tags = []
with open('reference_solution.csv', 'r') as f:
    lines = f.readlines()
    breed = lines[0].strip().split(',')[1:]
    lines = lines[1:]
    
    
    logits = []
    for line in lines:
        line = line.strip().split(',')
        tags.append(line[0])
        logits.append(list(map(float, line[1:])))
        
        
logits = np.array(logits)
print(logits.shape)
# get test labels

max_index = np.argmax(logits, axis=1)

# mapping labels

breed_map = {tags[i]: breed[max_index[i]] for i in range(len(tags))}

# write to labels_test.csv
with open('labels_test.csv', 'w') as f:
    f.write('id,breed\n')
    for tag in tags:
        f.write(f'{tag},{breed_map[tag]}\n')