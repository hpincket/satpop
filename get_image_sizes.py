from collections import defaultdict
from os.path import isfile, join
from os import listdir
import struct
import sys


def get_image_info(data):
    if is_png(data):
        w, h = struct.unpack('>LL', data[16:24])
        width = int(w)
        height = int(h)
    else:
        raise Exception('Not a PNG')

    return width, height


def is_png(data):
    return (data[:8] == b'\x89PNG\r\n\x1a\n' and (data[12:16] == b'IHDR'))


if __name__ == '__main__':
    file_names = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
    images_by_size = defaultdict()

    for file_name in file_names:
        with open(join(sys.argv[1], file_name), 'rb') as f:
            data = f.read()
            width, height = get_image_info(data)
            assert(isinstance(width, int))
            assert(isinstance(height, int))
            images_by_size[file_name] = (width, height)

    sorted_images = sorted(images_by_size.items(), key=lambda t: t[1][0] * t[1][1], reverse=True)

    print(sorted_images[:10])  # ten largest
    print(sorted_images[-10:])  # ten smallest
