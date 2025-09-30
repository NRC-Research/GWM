! BUTTERWORTH in F95 
! JRN, 12/2012
!  butterworth (fcin, fcout, df, fmin=0, fmax=0)
!      fcin     - input Fourier component 
!      fcout    - output Fourier component
!      df       - delta f for fcin and fcout
!      fmin     - lower corner frequency
!      fmax     - upper corner frequency 
! 
!      fmin = 0  : Low pass/High cut BUTTERWORTH filter
!      fmax = 0  : High pass/low cut BUTTERWORTH filter
!      both /= 0 : bandwidth filter. NOT IMPLEMENTED YET.
!      both = 0  : INVALID
!
SUBROUTINE butterworth (n, fcin, fcout, df, bwfmin, bwfmax, order)
    IMPLICIT NONE
    integer, intent(in) :: n
    integer, intent(IN) :: order  ! the order of the Butterworth filter
    complex, Dimension(n), intent(in) :: fcin  
    complex, Dimension(n), intent(out) :: fcout
    real, intent(in) :: df ! delta f, Hz
    REAL, INTENT(in) :: bwfmin ! in Hz 
    REAL, OPTIONAL :: bwfmax  ! in Hz
    
    !f2py complex intent(in) :: fc
    !f2py complex intent(out) :: fcout
    !f2py intent(in) :: df  
    !f2py real optional :: bwfmin = 0.0
    !f2py real optional :: bwfmax = 0.0
    !f2py integer optional :: order = 4
    !f2py integer intent(hide), depend(fcin) :: n=len(fcin)
    
    ! local variables
    REAL :: po, fo, pop, p, freq
    INTEGER :: i, o2 
    
    o2 = 2.0*order

    IF (bwfmin > 0.0) THEN 
!!$        print*, "HIGH PASS/LOW CUT BUTTERWORTH FILTER" 
        po = 0.5
        !~ fo = bwfmin*(po*po/(1.-po*po))**(1./8.)
        !~ pop = (4/bwfmin)*(bwfmin/fo)**9/(1.+(bwfmin/fo)**8)**1.5
        fo = bwfmin*(po*po/(1.-po*po))**(1./o2)
        pop = (4/bwfmin)*(bwfmin/fo)**(o2+1)/(1.+(bwfmin/fo)**o2)**1.5

        fcout(1) = 0.0
        highpass: DO i = 2, n
            freq = (i-1)*df
            !~ p = 1. / SQRT(1. + (bwfmin / freq)**8)
            p = 1. / SQRT(1. + (bwfmin / freq)**o2)
            IF (p < 0.5) p = MAX(po - pop * ( fo - freq), 0.0)
            fcout(i) = fcin(i) * p
        END DO highpass
    END IF
    
    IF ( bwfmax > 0.0) THEN
!!$        print*, "LOW PASS/HIGH CUT BUTTERWORTH FILTER" 
        lowpass: DO i = 1,n
            freq = (i-1) * df
            !~ p = 1. / SQRT(1. + (freq / bwfmax)**8)
            p = 1. / SQRT(1. + (freq / bwfmax)**o2)
            fcout(i) = fcin(i) * p
        END DO lowpass
    END IF 
    
    ! if both bwfmin and bwfmax == 0, then butterworth will do nothing.
!!$    
!!$    elseif (bwfmin /= 0.0 .and. bwfmax /=0.0) then
!!$        print*,  "Bandwidth filter, NOT IMPLEMENTED YET"
!!$        continue
!!$    else
!!$        print*, "fmin and fmax cannot be 0 at the same time &
!!$                &this branch is hold for possible error message"
!!$    end if
END SUBROUTINE butterworth

