import requests
s = requests
data={"sentence":"我是一个粉刷匠，粉刷本领强","keylist":"粉刷 粉墙,本领","sublist":"粉刷匠,本领强"}
r = s.post('http://10.5.190.22:8801/detect', data)
print (r.status_code)
print (r.headers['content-type'])
print (r.encoding)
print (r.text)
