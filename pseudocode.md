    import tensorflow
    import pytorch
    import jax
    import numpy
    import time

    n = ?

    matrix1 = np.random.uniform(low,high,size)
    matrix2 = np.random.uniform(low,high,size)

    tf_starttime = time.time()
    tf.matmul(matrix1,matrix2)
    tf_endtime = time.time()
    tf_total = tf_endtime - tf_starttime

    pt_starttime = time.time()
    pt.matmul(matrix1,matrix2)
    pt_endtime = time.time()
    pt_total = pt_endtime - pt_starttime

    j_starttime = time.time()
    j.matmul(matrix1,matrix2)
    j_endtime = time.time()
    j_total = j_endtime - j_starttime

    numpy_starttime = time.time()
    numpy.matmul(matrix1,matrix2)
    numpy_endtime = time.time()
    numpy_total = numpy_starttime - numpy_endtime

    print(tf_total)
    print(pt_total)
    print(j_total)
    print(numpy_total)