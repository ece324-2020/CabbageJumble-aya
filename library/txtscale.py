import re
import os

def resize_txt(path1, org_width = 3000, rszd_width  = 2048):
	''' This function takes in a path to a folder with the txt files and returns a text file with the resized values 
	     adjust the original width/height and resized width/height (width= height anyways) accordingly'''
	count = 1
	files = os.listdir(path1)
	files1 = [f for f in files if f.endswith('.txt')]
	for file1 in files1:
		toresize = []
		path = os.path.join(path1, file1)
		with open(path,'r',errors='ignore') as f:
			for line in f.readlines():
				as_list = re.split("\t|\n", line)
				print(as_list)
				toresize += [(int( int(as_list[0])*rszd_width/org_width), int(rszd_width/org_width*int(as_list[1])), int(float(as_list[2])*rszd_width/org_width))]

		with open(path, 'w') as f2:
			print(toresize)
			for i in range(len(toresize)):
				f2.write(f"{int(toresize[i][0])}\t{int(toresize[i][1])}\t{int(toresize[i][2])}\n")
		print(f"the converted path is {file1}\t | {count}/{len(files1)} txt files converted")
		count +=1
	return


if __name__ == "__main__":
	path1 = '/Users/yvielcastillejos/Desktop/Sample'
	resize_txt(path1)
