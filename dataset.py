from torch.utils.data import Dataset
from torch import tensor
import struct, binascii
class PE(Dataset):
    def __init__(self, fp_list, first_n_byte=2000000):
        """
        :param fp_list: list of strings, each is file-path.
        :param l2i: dict that maps label to index.
        :param first_n_byte: number of bytes to read from each file.
        """
        self.fp_list = fp_list
        # self.l2i = l2i ## no use
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    @staticmethod
    def represent_bytes(bytes_str):
        """
        :param bytes_str: string of bytes, i.e. two characters, each is one of 0-f.
        :return: integer, number between 0 to 257.
        """
        
        if bytes_str == '??':  # ignore those signs
            return 0
        return int(bytes_str, 16) + 1
        # return struct.unpack("h", bytes_str)[0] + 1

    def __getitem__(self, idx):
        with open(self.fp_list[idx][0], 'rb') as f:

            # print(self.first_n_byte)
            # byte = f.read(self.first_n_byte)
            # hexadecimal = binascii.hexlify(byte)
            # hexadecimal = [hexadecimal[i:i+2] for i in range(0, len(hexadecimal), 2)]
            bytes_array = np.array(bytearray(f.read()), dtype="uint8")
            
            # padding with zeroes such that all files will be of the same size
            if len(bytes_array) > self.first_n_byte:
                bytes_array = bytes_array[:self.first_n_byte]
            else:
                bytes_array = bytes_array + [0] * (self.first_n_byte - len(bytes_array))
            # print(len(tmp))
        f.close()
        return tensor(bytes_array), tensor(self.fp_list[idx][1])


class PE_Dataset(PE):
    def __init__(self, fp_list, first_n_byte=2000000):
        """
        :param fp_list: list of strings, each is file-path.
        :param first_n_byte: number of bytes to read from each file.
        """
        super(PE_Dataset, self).__init__(fp_list, first_n_byte)

    def __getitem__(self, idx):
        x, label = super(PE_Dataset, self).__getitem__(idx)
        # print(label)
        return x, tensor([label])