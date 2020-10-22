


# error local mapping
test_1: ../nptl/pthread_mutex_lock.c:79: __pthread_mutex_lock: Assertion `mutex->__data.__owner == 0' failed.

-> caused by delete map points

-> when local mapping
 a point can be deleted by the current process

-> need to reconsider the deletion of the points
