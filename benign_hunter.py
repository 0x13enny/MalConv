
import requests
from lxml import etree
import re, os

for page in range(1000000,1000001):
    url = "https://download.cnet.com/s/software/windows/?licenseType=Free&page=%s" %(page)
    resp = requests.get(url)
    html = resp.text
    selector=etree.HTML(html) 
    ret = selector.xpath("//*[@id=\"search-results\"]/a") #
    for obj in ret:
        resp_2 = requests.get("https://download.cnet.com" + re.sub('/3000-','/3001-',obj.attrib['href']))
        selector_2 = etree.HTML(resp_2.text)
        A = selector_2.xpath("//*[@id=\"pdl-manual\"]")
        for o in A:
            # print(o.attrib['href'])
            os.system("wget -O tmp "+re.sub("&","\&",o.attrib['href'])+" ; mv tmp $(md5sum tmp | cut -d' ' -f1)")
            # os.system(" "+re.sub("&","\&",o.attrib['href'])+" ; ")
        # break
            # print("wget -O tmp "+o.attrib["data-download-now-url"]+"; mv tmp $(md5sum tmp | cut -d' ' -f1)")
        #os.system("wget -i download.cnet.com" + re.sub('/3000-','/3001-',obj.attrib['href']))
	
        
# wget -O tmp https://files.downloadnow.com/s/software/16/75/57/53/DriverEasy_Setup.exe\?token\=1584540665_01285887793004e3210c56ef169225bc\&fileName\=DriverEasy_Setup.exe ; mv tmp $(md5sum tmp | cut -d' ' -f1)