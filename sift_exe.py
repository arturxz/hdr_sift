import os
import sys

def runDoG(dogFileName, imagePath, imageName, imageExtension):
	#compile file
	caminho_completo = actual_filepath +"/" +dogFileName
	os.system(f"g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename {caminho_completo}.cpp .cpp` {caminho_completo}.cpp `pkg-config --libs opencv`")
	
	#run file
	image = f"{imagePath}{imageName}.{imageExtension}"
	os.system(f"./{dogFileName} {image}")
	#os.system(f"cp ./*{dogFileName}*.txt {actual_filepath}/temp"  )

def runSift(dogFile, imagePath, imageName, imageExtension):
	# run keypoint orientation 
	keypointsFile = f"{imageName}.{dogFile}"
	image = f"{imagePath}{imageName}.{imageExtension}"
	currFilePath = actual_filepath
	print( " --- currFilePath ", currFilePath )
	print(f"calling: python3 {currFilePath}/siftKPOrientation.py {image} {imagePath} {keypointsFile}")
	os.system(f"python3 {currFilePath}/siftKPOrientation.py {image} {imagePath} {keypointsFile}")
	
	orientationFile = f"{keypointsFile}.kp.txt"
	print(f"calling: python3 {currFilePath}/siftDescriptor.py {orientationFile} {image}")
	os.system(f"python3 {currFilePath}/siftDescriptor.py {orientationFile} {image}")

"""
	MAIN 

	inputs: 1 - dogFile: string with the DoG type file
			2 - imagePath: string with the image's path
			3 - imageName: string with the image's name
			4 - imageExtension (optional): string with the image extension
"""
# LOCALIZACAO ATUAL DO ARQUIVO DESTE ARQUIVO
actual_filepath = os.path.dirname( os.path.realpath(__file__) )
print(actual_filepath)

# INICIALIZANDO VARIAVEIS
dogFile			= sys.argv[1]
imagePath		= "" 
imageName		= "" 
imageExtension	= ""

# TESTA SE TODOS OS ARGUMENTOS FORAM ENVIADOS E INFERE OS CAMINHOS SE POSSÍVEL
if( len( sys.argv ) <= 4 ):
	aux = sys.argv[2]
	aux = aux.split( "." )

	imageExtension = aux.pop()
	aux = "".join( aux[::-1] )

	aux = sys.argv[2]
	aux = aux.split( "/" )
	imageName = aux.pop()
	aux = imageName.split( "." )
	imageName = aux[0]

	aux = sys.argv[2]
	aux = aux.split( "/" )
	
	imagePath = "./"

	print( " ######## Path:", imagePath, "Ext:", imageExtension, "Name:", imageName )
else:
	print( sys.argv )

	imagePath = sys.argv[2]
	imageName = sys.argv[3]
	imageExtension = sys.argv[4]

runDoG(dogFile, imagePath, imageName, imageExtension)
runSift(dogFile, imagePath, imageName, imageExtension)
