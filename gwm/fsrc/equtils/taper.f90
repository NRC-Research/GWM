
! splitcosinebell
! return a split Cosine bell taper array for tapering a raw time history 
! record to minimize energy leackage. 
SUBROUTINE splitcosinebell(n, p, taper)
    IMPLICIT NONE
    INTEGER, intent(in) :: n
    REAL, intent(in) :: p  ! taper ratio
    REAL, DIMENSION(n), INTENT(out) :: taper
    !f2py intent(in) :: n
    !f2py real intent(out), depdend(n) :: taper
    !f2py real optional, intent(in) :: p=0.1  ! default is 10% of n
    
    INTEGER nt, i
    REAL, PARAMETER :: PI=3.141592653589793238462643383279502884197

    nt = NINT(n*p/2)  ! p/2 for each side
   
    ! left part
    DO i=1, nt
        taper(i) = 0.5*(1.0-COS(PI*(i-0.5)/nt))
    END DO
    
    ! middle part
    taper(nt+1:n-nt) = 1.0
    
    ! right part
    DO i=n+1-nt, n
        taper(i) = 0.5*(1.0-COS(PI*(n - i + 0.5)/nt))
    END DO
END SUBROUTINE splitcosinebell

