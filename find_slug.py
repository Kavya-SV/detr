import urllib.request
import re
req = urllib.request.Request("https://www.kaggle.com/search?q=bdd100k+weather", headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
matches = re.findall(r'/datasets/([^/]+/[^/\"\'?>]+)', html)
print(list(set(matches)))
