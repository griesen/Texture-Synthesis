import os
import bs4

os.system("wget http://www.cns.nyu.edu/lcv/texture/color/")

soup = bs4.BeautifulSoup(open('index.html'))

A = soup.findAll('center')[0].findAll('a')

for a in A:
    url = "http://www.cns.nyu.edu/lcv/texture/color/" + a.get('href').split('=')[-1].split("'")[0] + '.o.jpg'
    os.system('wget %s' % url) 
