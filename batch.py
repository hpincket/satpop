import csv
import random
import threading
from Queue import Queue, Empty, Full
from os.path import join
from threading import Event

import numpy as np
import png

import constants as C
from metadata_utils import generate_even_divisions


class BucketLabelTransformer():
    def __init__(self, bucket_maxes):
        self.bucket_maxes = bucket_maxes

    def transform_label(self, pop):
        for i, max in enumerate(self.bucket_maxes):
            if pop < max:
                return i
        return len(self.bucket_maxes)

    def number_of_labels(self):
        return len(self.bucket_maxes)


class RecordLoadingThread(threading.Thread):
    '''
    A Thread which consumes a line from the data.txv file from a queue
    and produces a data, label tuple and adds that to a another queue
    '''

    def __init__(self, threadID, stop_event, image_dimension, image_dir, task_queue, result_queue):
        super(RecordLoadingThread, self).__init__()
        self.threadID = threadID
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.image_dir = image_dir
        self.image_dimension = image_dimension
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            # Continually complete tasks
            row_task = self.task_queue.get(block=True, timeout=5)
            try:
                row_result = self.__process_row(row_task)
                self.result_queue.put(row_result, block=True, timeout=10)
            except IOError as e:
                pass
            except Empty as e:
                pass
            except Full as e:
                pass


    def __process_img(self, fname, dim):
        with open(fname, "rb") as imgfd:
            r = png.Reader(file=imgfd)
            h, w, pixels, metadata = r.read()
            print(h, w)
            assert (h == w)
            if metadata['alpha']:
                ans = []
                for lpixels in pixels:
                    ans.extend([v for i, v in enumerate(lpixels) if ((i + 1) % 4) != 0 or i == 0])
                image_1d = np.array(list(map(np.uint8, ans)))
            else:
                image_2d = np.vstack(map(np.uint8, pixels))
                image_1d = np.reshape(image_2d, (h * h * 3))
            if dim == 1:
                ret_image = image_1d
            elif dim == 3:
                ret_image = np.reshape(image_1d, (h, h, 3))
            else:
                raise ("Invalid image dimension")
            return ret_image

    def __process_row(self, row):
        image_filename = join(self.image_dir, "{}.{}".format(row[0], "png"))
        ret_image = self.__process_img(image_filename, self.image_dimension)
        return (ret_image, float(row[3]))


class SatPopBatch:
    '''
    The image/label iterator. Should be used inside a with ParallelSatPopBatcher
    '''

    def __init__(self,
                 data_fname,
                 batch_size,
                 task_queue,
                 result_queue,
                 label_transformer,
                 random=False
                 ):
        self.data_fname = data_fname
        self.batch_size = batch_size
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.offset = 0
        self.label_transformer = label_transformer
        self.random = random

    def __add_tasks_from_random_offset(self, num):
        self.offset = random.randint(0, 40000)
        self.__add_tasks(self.batch_size)

    def __add_tasks(self, num):
        if not self.random and self.offset > 42000:
            raise StopIteration
        with open(self.data_fname, "r") as data:
            tsvreader = csv.reader(data, delimiter='\t', quotechar='|')
            for i, row in enumerate(tsvreader):
                if i < self.offset:
                    continue
                if i > self.offset + num:
                    break
                self.task_queue.put(row)
            self.offset += num

    def __make_one_hot(self, dense_labels, max):
        one_hots = np.zeros((len(dense_labels), self.label_transformer.number_of_labels()))
        for i, val in enumerate(dense_labels):
            one_hots[i, val] = 1
        return one_hots

    def __iter__(self):
        return self

    # python 2 -- so I've read
    def next(self):
        return self.__next__()

    # python 3
    def __next__(self):
        '''
        Returns a batch of data and labels
        :return: ([serialized_imgs], [labels])
        '''
        # First, add more tasks to the task_queue if necessary
        print("Tasks: {}\tImgs: {}".format(self.task_queue.qsize(), self.result_queue.qsize()))
        while self.task_queue.qsize() < 500:
            if self.random:
                self.__add_tasks_from_random_offset(self.batch_size)
            else:
                self.__add_tasks(500)
        # Second, get the desired number of results out of the results queue
        serialized_imgs = []
        labels = []
        while len(labels) < self.batch_size:
            row_data, row_label = self.result_queue.get(block=True, timeout=None)
            serialized_imgs.append(row_data)
            labels.append(row_label)
        dense_labels = np.array([self.label_transformer.transform_label(l) for l in labels])
        labels = self.__make_one_hot(dense_labels, self.label_transformer.number_of_labels())
        return serialized_imgs, labels


class ParallelSatPopBatch:
    '''
    A python 'with' object with enter and exit functions.
    The enter() function returns a SatPopBatch
    '''

    def __init__(self, data_fname, image_dir, batch_size=20, random=False, image_dimension=1, label_transformer=None):
        self.data_fname = data_fname
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.random = random
        self.image_dimension = image_dimension
        if label_transformer:
            self.label_transformer = label_transformer
        else:
            self.label_transformer = BucketLabelTransformer(generate_even_divisions(10))
        self.task_queue = Queue(maxsize=1000)
        self.result_queue = Queue(maxsize=3000)  # This must be smaller because each item takes up more space
        self.stop_event = Event()  # Used to signal termination

    def __enter__(self):
        self.threads = self.__start_worker_process(4)
        return SatPopBatch(self.data_fname, self.batch_size,
                           self.task_queue, self.result_queue,
                           self.label_transformer,
                           random=self.random)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Inside the close")
        self.stop_event.set()
        for thread in self.threads:
            thread.join()

    def __start_worker_process(self, num):
        threads = []
        for i in range(num):
            thread = RecordLoadingThread(i, self.stop_event, self.image_dimension, self.image_dir, self.task_queue,
                                         self.result_queue, )
            thread.start()
            threads.append(thread)
        return threads


if __name__ == "__main__":
    pspb = ParallelSatPopBatch(C.SATPOP_MAIN_DATA_FILE, C.SATPOP_IMAGE_FOLDER)
    with pspb as spb:
        for batch in spb:
            print(len(batch[1]))
