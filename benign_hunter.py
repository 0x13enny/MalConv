
import requests
from lxml import etree
import re, os

for page in range(1):
    url = "https://download.cnet.com/s/software/windows/?page=%s" %(page)
    resp = requests.get(url)
    html = resp.text
    selector=etree.HTML(html) 
    ret = selector.xpath("//*[@id=\"search-results\"]/a") #
    for obj in ret:
        os.system("wget -i download.cnet.com" + re.sub('/3000-','/3001-',obj.attrib['href']))

        
