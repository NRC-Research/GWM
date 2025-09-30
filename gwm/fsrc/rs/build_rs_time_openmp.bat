f2py -c --compiler=mingw32 --fcompiler=gnu95 --f90flags="-fopenmp" -lgomp -lpthread -m _rs_time_openmp exactmethod_time_openmp.f90 

REM f2py exactmethod_time_openmp.f90 -m _rs_time_openmp -h rs_time_openmp.pyf
REM ~ f2py.py -c --compiler=mingw32 --fcompiler=gnu95 --f90flags="-fopenmp-simd" -lgomp -lpthread -m _rs_openmp exactmethod_openmp.f90

