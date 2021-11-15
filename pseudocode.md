    import tensorflow
    import pytorch
    import jax
    import numpy
    import time

    n = ?

    matrix1 = np.random.uniform(low,high,size)
    matrix2 = np.random.uniform(low,high,size)

    for(i = 0, i < num_tests; i++):
        tf_starttime = time.time()
        tf.matmul(matrix1,matrix2)
        tf_endtime = time.time()
        tf_total += tf_endtime - tf_starttime

    for(i = 0, i < num_tests; i++):
        pt_starttime = time.time()
        pt.matmul(matrix1,matrix2)
        pt_endtime = time.time()
        pt_total += pt_endtime - pt_starttime

    for(i = 0, i < num_tests; i++):
        j_starttime = time.time()
        j.matmul(matrix1,matrix2)
        j_endtime = time.time()
        j_total += j_endtime - j_starttime

    for(i = 0, i < num_tests; i++):
        numpy_starttime = time.time()
        numpy.matmul(matrix1,matrix2)
        numpy_endtime = time.time()
        numpy_total += numpy_starttime - numpy_endtime

    tf_avg = tf_total / num_tests
    pt_avg = py_total / num_tests
    j_avg = j_total / num_tests
    numpy_avg = numpy_total / num_tests

    print(tf_avg)
    print(pt_avg)
    print(j_avg)
    print(numpy_avg)