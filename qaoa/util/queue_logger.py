import multiprocessing as mp
import csv

class QueueLogger(object):
    """
    Records data from a multiprocessing.Queue to an output file

    Attributes
    ----------

    filename : string
        Name of the output file to write to
    buffersize : unsigned int
        Number of results to collect from the queue before writing them to file
    buffer : list 
        Used to cache results as read from the queue so that the output file need
        not be reopened for writing after each Queue.get() call
    count : unsigned int
        Number of times the write method as been called. Used to set the write mode
        to create a file on the initial call and append to it for subsequent calls. 
     
    """    

    def __init__(self,filename,quantities,buffersize,total):
        """
        Create a QueueLogger object that will read from a multiprocessing.Queue,
        store results in a buffer, and write the buffer to an output file when
        it is full or the queue is empty. A watcher process must be created in order
        to use this object.

        Parameters
        ----------
        filename : string
            Name of the output file to write to
        quantities : list of strings
            Names of quantities to be evaluated and recorded
        buffersize : unsigned int
            Number of results to collect from the queue before writing them to file
        total : unsigned int
            Number of items expected in the queue. Used for reporting progress.
        Example
        -------
        
        >>> import multiprocessing as mp
        >>> ql = QueueLogger('output.txt',50)
        >>> qresults = mp.Queue()
        >>> watcher = mp.Process(target=ql.read,args=(qresults,))
        >>> watcher.start()

        # Run some processes that write to qresults

        >>> watcher.terminate()

        """

        self.filename = filename
        self.buffersize = buffersize
        self.buffer = list()
        self.count = 0
        self.progress = 0
        self.total = total

        with open(self.filename,'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(quantities)

    def read(self,q):
        """
        Read results from a queue and append them to the buffer. Write the buffer
        to a file when full or the queue is empty

        Parameters
        ----------
        q : multiprocessing.Queue

        """

        print("\n\nCompute Progress:")
        print("0.00%")

        while True:
            while not q.empty():
                result = q.get()
                self.buffer.append(result)
                if len(self.buffer) >= self.buffersize:
                    self.progress += len(self.buffer)  
                    self.write_buffer()
                    print("{:.2%}".format(self.progress/self.total))

    def write_buffer(self):
         with open(self.filename,'a') as csvfile:
             writer = csv.writer(csvfile)
             while len(self.buffer):
                 b = self.buffer.pop()
                 if b is not None:
                     writer.writerow(b)
         self.count += 1
            

    
        
