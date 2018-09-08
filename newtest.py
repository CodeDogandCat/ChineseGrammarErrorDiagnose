f=open('./TNewsSegafter1.txt','r')
taihaole=open("./TNewsSegafter2.txt", "w")
breader=f.readlines()
for i in range(len(breader)):
    wenben=breader[i].replace(' ','')
    tem=' '.join(wenben)
    taihaole.write(tem)
taihaole.close()
f.close()
