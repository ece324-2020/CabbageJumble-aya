import os




labels = os.listdir("temp_labels")


dictionary = {(1, 72): 0, (1, 84): 1, (5, 72): 2, (5, 84): 3, (10, 72): 4, (10, 84): 5, (25, 72): 6, (25, 84): 7, (100, 72): 8, (100, 84): 9, (200, 72): 10, (200, 84): 11}


for i in labels:
    f = open(f"temp_labels/{i}","r")
    lines = f.read()
    print(lines)
    #for idx,j in enumerate(lines):
    #    print(j)
        #print(info)
