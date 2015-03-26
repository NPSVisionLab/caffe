
SET (CAFFE_LIBRARY
    ${CVAC_ROOT_DIR}/3rdparty/caffe/lib/libcaffe.a
    CACHE FILEPATH "Main caffe library" 
    )
SET (CAFFE_INCLUDE
    ${CVAC_ROOT_DIR}/3rdparty/caffe/include
    CACHE PATH "Main caffe include dir" 
    )
SET (CAFFE_CU_LIBRARY
    ${CVAC_ROOT_DIR}/3rdparty/caffe/lib/libcaffe_cu.a
    CACHE FILEPATH "Main caffe cu library" 
    )
SET (CAFFE_PROTO_LIBRARY
    ${CVAC_ROOT_DIR}/3rdparty/caffe/lib/libproto.a
    CACHE FILEPATH "Caffe proto library" 
    )
SET (CAFFE_BOOST_SYSTEM_LIBRARY
    /home/trbatcha/tools/lib/libboost_system.so
    CACHE FILEPATH "Boost System" 
    )
SET (CAFFE_BOOST_THREAD_LIBRARY
    /home/trbatcha/tools/lib/libboost_thread.so
    CACHE FILEPATH "Boost Thread" 
    )
SET (CAFFE_PROTOBUF_LIBRARY
    /home/trbatcha/tools/usr/lib/libprotobuf.so
    CACHE FILEPATH "Google protobuf library" 
    )
SET (CAFFE_PROTOBUFLITE_LIBRARY
    /home/trbatcha/tools/usr/lib/libprotobuf-lite.so
    CACHE FILEPATH "Google protobuf-lite library" 
    )
SET (CAFFE_GOOGLELOG_LIBRARY
    /home/trbatcha/tools/usr/lib/libglog.so
    CACHE FILEPATH "Google Log library" 
    )
SET (CAFFE_GOOGLEFLAGS_LIBRARY
    /home/trbatcha/gflags-2.1.1/build/lib/libgflags.a
    CACHE FILEPATH "Google Flags library" 
    )
SET (CAFFE_GOOGLEFLAGS_INCLUDE
    /home/trbatcha/gflags-2.1.1/build/include
    CACHE FILEPATH "Google Flags include dir" 
    )
SET (CAFFE_LEVELDB_LIBRARY
    /home/trbatcha/leveldb-master/libleveldb.so
    CACHE FILEPATH "LevelDB library" 
    )
SET (CAFFE_LEVELDB_INCLUDE
    /home/trbatcha/leveldb-master/include
    CACHE FILEPATH "LevelDB include dir" 
    )
SET (CAFFE_LMDB_LIBRARY
    /home/trbatcha/liblmdb/liblmdb.so
    CACHE FILEPATH "LMDB library" 
    )
SET (CAFFE_LMDB_INCLUDE
    /home/trbatcha/liblmdb
    CACHE PATH "LMDB include dir" 
    )
SET (CAFFE_HDF5_LIBRARY
    /home/trbatcha/tools/usr/lib64/libhdf5.so
    CACHE FILEPATH "HDF5 library" 
    )
SET (CAFFE_HDF5HL_LIBRARY
    /home/trbatcha/tools/usr/lib64/libhdf5_hl.so
    CACHE FILEPATH "HDF5 High Level library" 
    )
SET (CAFFE_SNAPPY_LIBRARY
    /home/trbatcha/usr/lib/libsnappy.so
    CACHE FILEPATH "Snappy library" 
    )
SET (CAFFE_BLAS_LIBRARY
    /usr/lib64/libblas.so
    CACHE FILEPATH "Blas library" 
    )
SET (CAFFE_BLAS_INCLUDE
    /usr/include/gsl
    CACHE FILEPATH "Blas include dir" 
    )
SET (CAFFE_CBLAS_LIBRARY
    /usr/lib64/atlas/libcblas.so.3
    CACHE FILEPATH "CBlas library" 
    )
SET (CAFFE_LAPACK_LIBRARY
    /usr/lib64/atlas/liblapack.so.3
    CACHE FILEPATH "LaPack library" 
    )
SET (CAFFE_CUDART_LIBRARY
    /usr/local/cuda-6.5/lib64/libcudart.so
    CACHE FILEPATH "Cuda RT library" 
    )
SET (CAFFE_CUDA_INCLUDE
    /usr/local/cuda-6.5/include
    CACHE PATH "Cuda include dir" 
    )
SET (CAFFE_CUDARAND_LIBRARY
    /usr/local/cuda-6.5/lib64/libcurand.so
    CACHE FILEPATH "Cuda Rand library" 
    )
SET (CAFFE_CUDASPARSE_LIBRARY
    /usr/local/cuda-6.5/lib64/libcusparse.so
    CACHE FILEPATH "Cuda Sparse library" 
    )
SET (CAFFE_CUDANPPC_LIBRARY
    /usr/local/cuda-6.5/lib64/libnppc.so
    CACHE FILEPATH "Cuda Nppc library" 
    )
SET (CAFFE_CUDANPPI_LIBRARY
    /usr/local/cuda-6.5/lib64/libnppi.so
    CACHE FILEPATH "Cuda Nppi library" 
    )
SET (CAFFE_CUDANPPS_LIBRARY
    /usr/local/cuda-6.5/lib64/libnpps.so
    CACHE FILEPATH "Cuda Npps library" 
    )
SET (CAFFE_CUDABLAS_LIBRARY
    /usr/local/cuda-6.5/lib64/libcublas.so
    CACHE FILEPATH "Cuda Blas library" 
    )
SET (CAFFE_CUDAFFT_LIBRARY
    /usr/local/cuda-6.5/lib64/libcufft.so
    CACHE FILEPATH "Cuda FFT library" 
    )
SET (CAFFE_CUDAPTI_LIBRARY
    /usr/local/cuda-6.5/extras/CUPTI/lib64/libcupti.so
    CACHE FILEPATH "Cuda PTI library" 
    )
SET (CAFFE_CUDA_LIBRARY
    /usr/lib64/libcuda.so
    CACHE FILEPATH "Cuda library" 
    )

SET (CAFFE_LIBRARIES
    ${CAFFE_LIBRARY}
    ${CAFFE_CU_LIBRARY}
    ${CAFFE_PROTO_LIBRARY}
    ${CAFFE_BOOST_SYSTEM_LIBRARY}
    ${CAFFE_BOOST_THREAD_LIBRARY}
    ${CAFFE_PROTOBUF_LIBRARY}
    ${CAFFE_PROTOBUFLITE_LIBRARY}
    ${CAFFE_GOOGLELOG_LIBRARY}
    ${CAFFE_GOOGLEFLAGS_LIBRARY}
    ${CAFFE_LEVELDB_LIBRARY}
    ${CAFFE_LMDB_LIBRARY}
    ${CAFFE_HDF5_LIBRARY}
    ${CAFFE_HDF5HL_LIBRARY}
    ${CAFFE_SNAPPY_LIBRARY}
    ${CAFFE_BLAS_LIBRARY}
    ${CAFFE_CBLAS_LIBRARY}
    ${CAFFE_LAPACK_LIBRARY}
    ${CAFFE_CUDART_LIBRARY}
    ${CAFFE_CUDARAND_LIBRARY}
    ${CAFFE_CUDASPARSE_LIBRARY}
    ${CAFFE_CUDANPPC_LIBRARY}
    ${CAFFE_CUDANPPI_LIBRARY}
    ${CAFFE_CUDANPPS_LIBRARY}
    ${CAFFE_CUDABLAS_LIBRARY}
    ${CAFFE_CUDAFFT_LIBRARY}
    ${CAFFE_CUDAPTI_LIBRARY}
    ${CAFFE_CUDA_LIBRARY}
    )

SET (CAFFE_INCLUDE_DIRS
     ${CAFFE_INCLUDE} 
     ${CAFFE_GOOGLEFLAGS_INCLUDE} 
     ${CAFFE_LEVELDB_INCLUDE}
     ${CAFFE_LMDB_INCLUDE}
     ${CAFFE_BLAS_INCLUDE}
     ${CAFFE_CUDA_INCLUDE}
    )
ADD_SUBDIRECTORY(detectorService)
#ADD_SUBDIRECTORY(trainerService)
