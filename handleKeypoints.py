class KeyPoint:
    def __init__(self, x, y, scale, dir):
        self.x = x
        self.y = y
        self.dir = dir
        self.scale = scale

    def setDir(self, dir):
        self.dir = dir

class KeyPointList:
    ## Make sure that absPath ends with '/' 
    # Add keypoint scale if it exist in file 
    def __init__(self, absPath, fileName):
        auxList = []
        completePath = absPath + fileName
        
        first = True
        f = open(completePath + "1.txt", "r")
        for line in f:
            if first:
                first = False
                continue
            
            y, x, v = line.split()

            kp = KeyPoint(int(x), int(y), 1, 0)
            auxList.append(kp)
        f.close()

        first = True
        f = open(completePath + "2.txt", "r")
        for line in f:
            if first:
                first = False
                continue
            
            y, x, v = line.split()

            kp = KeyPoint(int(x), int(y), 1, 0)
            auxList.append(kp)
        f.close()

        first = True
        f = open(completePath + "3.txt", "r")
        for line in f:
            if first:
                first = False
                continue
            
            y, x, v = line.split()

            kp = KeyPoint(int(x), int(y), 1 ,0)
            auxList.append(kp)
        f.close()

        self.List = auxList

#keypoints = KeyPointList('/home/welerson/Área de Trabalho/Pesquisa /dataset/2D/distance/', '100', 'HDR.surf')
