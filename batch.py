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
    '''
    Modifies populations to labels
    bucket_maxes is a list of values which define the upper bound on the buckets.
    '''
    def __init__(self, bucket_maxes):
        self.bucket_maxes = bucket_maxes

    def transform_label(self, pop):
        ''' Given a population, determine which bucket it belongs to'''
        for i, max in enumerate(self.bucket_maxes):
            if pop < max:
                return i
        return len(self.bucket_maxes)

    def number_of_labels(self):
        ''' The number of label options'''
        return len(self.bucket_maxes)


class RecordLoadingThread(threading.Thread):
    '''
    A Thread which consumes a line from the data.txv file from a queue
    and produces a data, label tuple and adds that to a another queue
    '''

    def __init__(self, threadID, stop_event, image_dimension, image_dir, task_queue, result_queue):
        '''
        Create a new thread
        :param threadID: a given number identifying this thread for debugging
        :param stop_event: used to signal an exit from the while loop
        :param image_dimension: 1 or 3. How the image should be processed
        :param image_dir:
        :param task_queue: where info from data.tsv is placed. This thread consumes.
        :param result_queue: where images and labels are placed. This thread produces.
        '''
        super(RecordLoadingThread, self).__init__()
        self.threadID = threadID
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.image_dir = image_dir
        self.image_dimension = image_dimension
        self.stop_event = stop_event

    def run(self):
        ''' Loop forever '''
        while not self.stop_event.is_set():
            # Continually complete tasks
            row_task = self.task_queue.get(block=True, timeout=5)
            try:
                row_result = self.__process_row(row_task)
                if row_result[0] is not None:  # Used for odd error case.
                    self.result_queue.put(row_result, block=True, timeout=10)
            except IOError as e:
                pass
            except Empty as e:
                pass
            except Full as e:
                pass


    def __process_img(self, fname, dim):
        ''' Open image, return it in the correct format '''
        with open(fname, "rb") as imgfd:
            r = png.Reader(file=imgfd)
            h, w, pixels, metadata = r.read()
            assert (h == w)
            if metadata['alpha']:
                # Lord knows why this is necessary
                ans = []
                for lpixels in pixels:
                    ans.extend([v for i, v in enumerate(lpixels) if ((i + 1) % 4) != 0 or i == 0])
                image_1d = np.array(list(map(np.uint8, ans)))
            else:
                image_2d = np.vstack(map(np.uint8, pixels))
                try:
                    image_1d = np.reshape(image_2d, (h * h * 3))
                except ValueError:
                    return None  # Will be ignored later.
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

    def __init__(self, data_fname, batch_size, task_queue, result_queue, label_transformer, random=False):
        '''
        Create a new iterator.
        Most of these items are passed on to worker threads.
        '''
        self.data_fname = data_fname
        self.batch_size = batch_size
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.offset = 0
        self.label_transformer = label_transformer
        self.random = random

    def __add_tasks_from_random_offset(self, num):
        ''' Used if random=True '''
        self.offset = random.randint(0, 40000)
        self.__add_tasks(self.batch_size)

    def __add_tasks(self, num):
        ''' Seeks to offset, reads num files'''
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
        max = 21
        if max == 3:
            y_0 = [0.75, 0.25, 0.0] # rural
            y_1 = [0.25, 0.5, 0.25] # surburban
            y_2 = [0.0, 0.25, 0.75] # urban
            
            for i, val in enumerate(dense_labels):
                if val == 0:
                    one_hots[i] = y_0
                elif val == 1:
                    one_hots[i] = y_1
                elif val == 2:
                    one_hots[i] = y_2
        elif max == 5:
            y_0 = [0.75, 0.2, 0.05, 0.0, 0.0] # unpopulated
            y_1 = [0.25, 0.5, 0.2, 0.05, 0.0] # rural
            y_2 = [0.05, 0.2, 0.5, 0.2, 0.05] # suburban
            y_3 = [0.0, 0.05, 0.2, 0.5, 0.25] # urban
            y_4 = [0.0, 0.0, 0.05, 0.2, 0.75] # highly urban

            for i, val in enumerate(dense_labels):
                if val == 0:
                    one_hots[i] = y_0
                elif val == 1:
                    one_hots[i] = y_1
                elif val == 2:
                    one_hots[i] = y_2
                elif val == 3:
                    one_hots[i] = y_3
                elif val == 4:
                    one_hots[i] = y_4
        else:
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
        # print("Tasks: {}\tImgs: {}".format(self.task_queue.qsize(), self.result_queue.qsize()))
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
