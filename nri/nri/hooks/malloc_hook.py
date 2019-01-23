import time
import traceback
import logging

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    raise ImportError('CuPy is not installed.')

if CUPY_AVAILABLE:

    class MallocHook(cupy.cuda.memory_hook.MemoryHook):

        def __init__(self):
            super(MallocHook, self).__init__()
            self.logger = logging.getLogger(__name__)

        def alloc_preprocess(self, device_id, mem_size):
            self.t = time.time()

        def alloc_postprocess(self, device_id, mem_size, mem_ptr):
            ms = (time.time() - self.t) * 1000
            if ms >= 1:
                tb = traceback.format_list(traceback.extract_stack())
                last_line = '|'.join([l.strip().replace('\n', '|') for l in tb[-3:-1]])
                self.logger.info('| {} ms | size {} | {}'.format(
                    ms, mem_size, last_line.strip()))

        def malloc_preprocess(self, device_id, size, mem_size):
            self.mt = time.time()

        def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
            ms = (time.time() - self.mt) * 1000
            if ms >= 1:
                tb = traceback.format_list(traceback.extract_stack())
                last_line = '|'.join([l.strip().replace('\n', '|') for l in tb[-3:-1]])
                self.logger.info('| (Pool) | {} ms | size {} | {}'.format(
                    ms, mem_size, last_line.strip()))

