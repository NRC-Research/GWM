f2py -c --compiler=mingw32 --fcompiler=gnu95 --f90flags="-fbounds-check -g" -m _equtils smooth.f90 taper.f90 butterworth.f90 zpa_clipping.f90
