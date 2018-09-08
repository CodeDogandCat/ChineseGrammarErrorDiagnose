from pycorrector import util
f1 = open('16_old.txt','r',encoding='utf-8')
f2 = open('16_new.txt','w',encoding='utf-8')
bline = f1.readlines()
for i in range(len(bline)):
    sent = util.traditional2simplified(bline[i])
    print(sent)
    f2.write(sent)
f1.close()
f2.close()
