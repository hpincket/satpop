from os import listdir, remove
from os.path import isfile, join
import sys

def is_png(data):
    return (data[:8] == b'\x89PNG\r\n\x1a\n' and (data[12:16] == b'IHDR'))

def is_html(data):
    return (data[:6] == b'<HTML>')

if __name__ == '__main__':
    '''
    argv[1] := directory containing the images
    '''
    img_dir = sys.argv[1]
    file_names = [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f))]
    for file_name in file_names:
        with open(file_name, 'rb') as fd:
            data = fd.read(16)
        if is_png(data):
            continue
        elif is_html(data):
            print("About to delete: {}".format(file_name))
            remove(file_name)
        else:
            print(data)
            exit(1)
    print("Done")

