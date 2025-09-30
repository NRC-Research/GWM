! smohth with a window of a percentage width and unity amplitude
! assuming an equal spacing of frequency points
SUBROUTINE rsmoothen_perc(arr, outarr, percwin, n)
    INTEGER, INTENT(in) :: n
    REAL, DIMENSION(n),intent(in) :: arr
    REAL, DIMENSION(n),intent(out) :: outarr
    REAL, INTENT(in) :: percwin
    !f2py intent(in) :: arr
    !f2py intent(out) :: outarr
    !f2py intent(in) :: percwin
    !f2py integer intent(hide), depend(arr) :: n=len(arr)
    
    INTEGER :: icomp, ilow, ihigh
    INTEGER :: nhalfwin

    ! first low freq is not smoothed        
    outarr(1) = arr(1)
    outarr(n) = arr(n)
    LOOPF0: DO icomp = 2, n - 1
        nhalfwin = nint( icomp * percwin /2.0 )
        ilow = icomp - nhalfwin
        ihigh = icomp + nhalfwin
        IF (ilow < 1) THEN
            ilow = 1
            ihigh = icomp + (icomp - ilow)
        ELSE IF (ihigh > n) THEN
            ihigh = n
            ilow = icomp - (n - icomp)
        END IF

        outarr(icomp) = sum(arr(ilow:ihigh)) / (ihigh-ilow+1)
       !  print*, ilow, icomp, ihigh, arr(icomp), outarr(icomp)
    END DO LOOPF0
    
END SUBROUTINE rsmoothen_perc


! Smooth
! smoothen a record with a window of a total unity weight
! W - half width,  n - lenght of data record
! the smoothing window is assumed to have been normalized
! real version
SUBROUTINE rsmoothen(arr, outarr, win, n, W)
    INTEGER, INTENT(in) :: n, W
    REAL, DIMENSION(n),intent(in) :: arr
    REAL, DIMENSION(n),intent(out) :: outarr
    REAL, DIMENSION(-W:W), INTENT(in) :: win
    !f2py intent(in) :: arr
    !f2py intent(out) :: outarr
    !f2py intent(in) :: win
    !f2py integer intent(hide), depend(arr) :: n=len(arr)
    !f2py integer intent(hide), depend(win) :: w=(len(win)-1)/2
    
    INTEGER i
    
    ! middle part
    DO i = W+1, n-W
        outarr(i) = DOT_PRODUCT(arr(i-W:i+W), win)
    END DO

    ! beginning and ending parts
    ! copy without smoothing. If need to smooth, do this in Python with each
    ! point with a new window
    outarr(1:W) = arr(1:W)
    outarr(n-W+1:n) = arr(n-W+1:n)
END SUBROUTINE rsmoothen

! complex version
SUBROUTINE csmoothen(arr, outarr, win, n, W)
    INTEGER, INTENT(in) :: n, W
    COMPLEX, DIMENSION(n),INTENT(in) :: arr
    COMPLEX, DIMENSION(n),INTENT(out) :: outarr
    REAL, DIMENSION(-W:W), INTENT(in) :: win
    !f2py intent(in) :: arr
    !f2py intent(out) :: outarr
    !f2py intent(in) :: win
    !f2py integer intent(hide), depend(arr) :: n=len(arr)
    !f2py integer intent(hide), depend(win) :: w=(len(win)-1)/2
    
    INTEGER i
    
    ! middle part
    DO i = W+1, n-W
        outarr(i) = DOT_PRODUCT(arr(i-W:i+W), win)
    END DO

    ! beginning and ending parts
    ! copy without smoothing. If need to smooth, do this in Python with each
    ! point with a new window
    outarr(1:W) = arr(1:W)
    outarr(n-W+1:n) = arr(n-W+1:n)
END SUBROUTINE csmoothen

! hamming window
! return a Hamming window with a half width W, and being normalized to have a 
! total weight of unity. The number of point in the window is 2*W + 1.
! For example, for a 11-point Hamming window, W=5.
SUBROUTINE hamming(W, win)
    IMPLICIT NONE
    INTEGER, INTENT(in) :: W
    REAL, DIMENSION(-W:W), INTENT(out) :: win

    REAL, PARAMETER :: PI=3.141592653589793238462643383279502884197
    INTEGER i

    DO i = -W, W
        win(i) = 0.53836 + 0.46164*COS(PI*i/W)
    END DO
    ! normalize
    win = win / SUM(win)
END SUBROUTINE hamming

! hann window
! return a Hann window with a half width W, and being normalized to have a 
! total weight of unity. The number of point in the window is 2*W + 1.
! For example, for a 11-point Hamming window, W=5.
SUBROUTINE hann(W, win)
    IMPLICIT NONE
    INTEGER, INTENT(in) :: W
    REAL, DIMENSION(-W:W), INTENT(out) :: win

    REAL, PARAMETER :: PI=3.141592653589793238462643383279502884197   
    INTEGER i

    DO i = -W, W
        win(i) = 0.5 + 0.5*COS(PI*i/W)
    END DO
    ! normalize
    win = win / SUM(win)
END SUBROUTINE hann

! triangular window
! return a Triangular window with a half width W, and being normalized 
! to have a total weight of unity. The number of point in the window 
! is 2*W + 1. For example, for a 11-point Hamming window, W=5.
SUBROUTINE triangular(W, win)
    IMPLICIT NONE
    INTEGER, INTENT(in) :: W
    REAL, DIMENSION(-W:W), INTENT(out) :: win

    INTEGER i, N

    N = 2*W + 1
    DO i = -W, W
        win(i) = N - 2*ABS(i)  ! 1, 3, 5, ..., N, ..., 5, 3, 1
    END DO
    ! normalize
    win = win / SUM(win)
END SUBROUTINE triangular





