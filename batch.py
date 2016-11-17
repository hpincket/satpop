import constants as C
from os.path import join
import csv
import png
import numpy

class SatPopBatch:

    def __init__(self, data_fname, image_dir, batch_size=20):
        self.offset = 0
        self.data_fname = data_fname
        self.image_dir = image_dir
        self.batch_size = batch_size

    def __process_row(self, row):
        image_filename = join(self.image_dir, "{}.{}".format(row[0], "png"))
        with open(image_filename, "rb") as imgfd:
            r = png.Reader(file=imgfd)
            h, w, pixels, metadata = r.read()
            assert(h == 512)
            assert(w == 512)
            if metadata['alpha']:
                ans = []
                for lpixels in pixels:
                    ans.extend([v for i,v in enumerate(lpixels) if ((i+1) % 4) != 0 or i == 0])
                image_1d = numpy.array(map(numpy.uint8, ans))
            else:
                image_2d = numpy.vstack(map(numpy.uint8, pixels))
                image_1d = numpy.reshape(image_2d, (512*512*3))
        return (image_1d ,float(row[3]))

    def __iter__(self):
        return self

    # python 2 -- so I've heard I read
    def next(self):
        self.__next__()

    # python 3
    def __next__(self):
        '''
        Returns a batch of data and labels
        :return: ([serialized_imgs], [labels])
        '''
        serialized_imgs = []
        labels = []
        with open(self.data_fname, "r") as data:
            tsvreader = csv.reader(data, delimiter='\t', quotechar='|')
            for i,row in enumerate(tsvreader):
                if i < self.offset:
                    continue
                if len(labels) >= self.batch_size:
                    break
                try:
                    row_data, row_label = self.__process_row(row)
                    serialized_imgs.append(row_data)
                    labels.append(row_label)
                except FileNotFoundError:
                    continue
        if len(labels) == 0:
            raise StopIteration
        self.offset += len(labels)
        return (serialized_imgs, labels)


if __name__ == "__main__":
    spb = SatPopBatch(C.SATPOP_MAIN_DATA_FILE, C.SATPOP_IMAGE_FOLDER)
    for batch in spb:
        print(len(batch[1]))

