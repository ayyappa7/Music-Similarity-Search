import numpy as np;
import os
musicPath="D:\MSS\TagATune\mp3";
annotation=open("D:\MSS\TagATuneUnZip\\annotations_final.csv","r")
line=annotation.readline()
lines=line.split("\t")
print(lines[0])
genreIndex=[]
genreIndex=np.append(genreIndex,int(lines.index("\"blues\"")))
genreIndex=np.append(genreIndex,int(lines.index("\"classical\"")))
genreIndex=np.append(genreIndex,int(lines.index("\"country\"")))
genreIndex=np.append(genreIndex,int(lines.index("\"disco\"")))
genreIndex=np.append(genreIndex,int(lines.index("\"hip hop\"")))
genreIndex=np.append(genreIndex,int(lines.index("\"jazz\"")))
genreIndex=np.append(genreIndex,int(lines.index("\"metal\"")))
genreIndex=np.append(genreIndex,int(lines.index("\"pop\"")))
genreIndex=np.append(genreIndex,int(lines.index("\"reggae\"")))
genreIndex=np.append(genreIndex,int(lines.index("\"rock\"")))
print(genreIndex)
annotationLines=annotation.readlines()
for line in annotationLines:
    words=line.split("\t")
    for word in words:
        word2=word.replace('\"','')
        words[words.index(word)]=word2
    fileName = words[np.size(words) - 1][:-1]
    fileName=fileName.replace("/","\\")
    i=0
    for a in genreIndex:
        if words[int(a)]== "1":
            print(" genre="+str(i))
            dest="D:\MSS\\allMusic\\"+str(i)+"\\"
            source="D:\MSS\TagATune\mp3\\"
            os.chdir(source)
            cmd="copy "+fileName+" "+dest
            print(cmd)
            os.system(cmd)
        i+=1



