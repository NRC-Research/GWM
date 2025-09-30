! zpa_clipping
SUBROUTINE clip_at_zpa(arr, zpa, outarr, clip_count, n)
    INTEGER, INTENT(in) :: n
    REAL, DIMENSION(n),intent(in) :: arr
    REAL, intent(in) :: zpa
    REAL, DIMENSION(n),intent(out) :: outarr
    integer, intent(out) :: clip_count
    !f2py intent(in) :: arr
    !f2py intent(in) :: zpa
    !f2py intent(out) :: outarr
    !f2py intent(out) :: clip_count
    !f2py integer intent(hide), depend(arr) :: n=len(arr)
    
    INTEGER i
    REAL zpa2
    
    zpa2 = 2*zpa
    clip_count = 0
    ! middle part
    DO i = 1, n
        if (arr(i) > zpa) then
            outarr(i) = zpa2 - arr(i)
            clip_count = clip_count + 1
        else if (arr(i) < -zpa) then
            outarr(i) = -zpa2 - arr(i)
            clip_count = clip_count + 1
        else
            outarr(i) = arr(i)
        end if
    END DO
END SUBROUTINE clip_at_zpa