from multiprocessing import Pool
import math
def runn(data, index, size):
    resstr=''
    startstr = '<start>'
    endstr = '<end>'
    size = math.ceil(len(data) / size)
    start = size * index
    end = (index + 1) * size if (index + 1) * size < len(data) else len(data)
    for i in range(start,end):
        if len(data[i].strip()) != 0:
            abc = data[i].strip().replace(' ', '').replace('\n', '').replace('\r\n', '').replace('\r', '').strip(
                '\r\n')
            strb = startstr + abc[0] + ' '
            for j in range(len(abc) - 1):
                strb = strb + abc[j] + abc[j + 1] + ' '
            strb = strb + abc[len(abc) - 1] + endstr + '\n'
        else:
            strb = ''
        resstr=resstr+strb
    return resstr
if __name__ == '__main__':
    f = open("./TNewsSeg.txt", "r", encoding='UTF-8')
    f2 = open("./TNewsSeg2.txt", "w", encoding='UTF-8')
    res=[]
    bline = f.readlines()
    processor = 40
    p = Pool(processor)
    for i in range(processor):
        res.append(p.apply_async(runn, args=(bline, i, processor,)))
    p.close()
    p.join()
    for i in res:
        f2.write(i.get())
