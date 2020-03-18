
import pandas as pd
import sys
import binascii
import matplotlib.pyplot as plt

A = pd.read_csv(sys.argv[1])
B = pd.read_csv(sys.argv[2])
C = pd.concat([A,B],ignore_index=True)

benign_len = []
malicious_len = []

for i in range(C.shape[0]):

    with open(C['path'][i], 'rb') as f:
        byte = f.read()
        hexadecimal = binascii.hexlify(byte)
        hexadecimal = [hexadecimal[i:i+2] for i in range(0, len(hexadecimal), 2)]
        if C['label'][i]==1:
            malicious_len.append(len(hexadecimal)/1024/1024) 
        else:
            benign_len.append(len(hexadecimal)/1024/1024) #Mega Bytes

plt.figure(1)
plt.hist(benign_len, label="benign file size (MB)")
plt.savefig('benign.png')
plt.figure(2)
plt.hist(malicious_len)
plt.savefig('malicious.png',label="benign file size (MB)")


