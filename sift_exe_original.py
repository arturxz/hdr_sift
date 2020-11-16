import os
import sys

# getting args passed: 1:dogFile; 2:imagePath (must ends with /) 3:imageName; 4:imageExtension

dogFile = sys.argv[1] 
imagePath = sys.argv[2] 
imageName = sys.argv[3] 
imageExtension = sys.argv[4] 

def runDoG(dogFileName, imagePath, imageName, imageExtension):
	#compile file
	os.system(f"g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename {dogFileName}.cpp .cpp` {dogFileName}.cpp `pkg-config --libs opencv`")
	
	#run file
	image = f"{imagePath}{imageName}.{imageExtension}"
	os.system(f"./{dogFileName} {image}")

def runSift(dogFile, imagePath, imageName, imageExtension):
	# run keypoint orientation 
	keypointsFile = f"{imageName}.{dogFile}"
	image = f"{imagePath}{imageName}.{imageExtension}"
	
	print(f"calling: python3 siftKPOrientation.py {image} {imagePath} {keypointsFile}")
	os.system(f"python3 siftKPOrientation.py {image} {imagePath} {keypointsFile}")
	
	orientationFile = f"{keypointsFile}.kp.txt"
	print(f"calling: python3 siftDescriptor.py {orientationFile} {image}")
	os.system(f"python3 siftDescriptor.py {orientationFile} {image}")

# MAIN #
runDoG(dogFile, imagePath, imageName, imageExtension)
runSift(dogFile, imagePath, imageName, imageExtension)
