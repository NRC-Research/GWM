! RS_Nigam: response spectrum calculation based on:
!     Digital Calculation of Response Spectra from Strong-motion Earthquake
!     Records
!     by Navin C. Nigam and Paul C. Jennings
!     EERL, CIT, Pasadena, California
!     June 1968.

! FOR MODIFICATION:
! Three places to check: f2py comments, omp comments, and the actual Fortran code

SUBROUTINE rst_exactmethod(nd, np, na, dmp, pd, ga, del, sf, &
           sd, sv, sa, ta, pa, fs)
    !f2py intent(in) :: dmp
    !f2py intent(in) :: pd
    !f2py intent(in) :: ga
    !f2py intent(in) :: del
    !f2py real, optional, intent(in) :: sf = 1.0
    !f2py intent(out) :: sd
    !f2py intent(out) :: sv
    !f2py intent(out) :: sa
    !f2py intent(out) :: ta
    !f2py intent(out) :: pa
    !f2py intent(out) :: fs
    !f2py integer, intent(hide), depend(dmp) :: nd=len(dmp)
    !f2py integer, intent(hide), depend(pd) ::  np=len(pd)
    !f2py integer, intent(hide), depend(ga) ::  na=len(ga)

    INTEGER, INTENT(in) :: nd, np, na  ! numbers of damping, periods, acc
    REAL, INTENT(in) :: dmp(nd)   ! dampings
    REAL, INTENT(in) :: pd(np)    ! periods (inverse of frequencies)
    REAL, INTENT(in) :: ga(na)    ! ground accelerogram
    REAL, INTENT(in) :: del       ! dt for accelerogram
    REAL, INTENT(in) :: sf        ! scale factor for accelerogram
    REAL, INTENT(out) :: sd(nd, np)  ! displacement spectrum
    REAL, INTENT(out) :: sv(nd, np)  ! velocity spectrum
    REAL, INTENT(out) :: sa(nd, np)  ! (Absolute) acceleration spectrum
    REAL, INTENT(out) :: ta(nd, np)  ! time for peak acceleration response
    REAL, INTENT(out) :: pa(nd, np)  ! sign of the peak acceleration response (+1.0 or -1.0)
    REAL, INTENT(out) :: fs(np)      ! (Absolute) acceleration spectrum

    ! local variables
    REAL :: x(3), g(2), A(2,2), B(2,2), ty(3), absty(3)
    INTEGER :: j, k, i, nadd, m
    INTEGER :: ni            ! number of integration interval (L)
    REAL :: d, p, w, delp, delt, dmax, vmax, amax, dw,  w2, sl
    INTEGER :: iamax  ! index of ga when amax response occurs
    REAL :: psign
    
    REAL :: vend
    REAL :: ga1, ga2
    REAL, PARAMETER :: UNDAMPED = 1.0E-3

    LDAMP: DO j=1, nd
        d = dmp(j)
        !$omp parallel default(private) &
        !$omp shared(nd, np, na, &
        !$omp      dmp, pd, ga, &
        !$omp      del, sf, &
        !$omp      sd, sv, sa, ta, pa, fs, &
        !$omp      j, d) &
        !$omp PROC_BIND(SPREAD)
        !$omp do &
        !$omp schedule(static) 
        LPERIOD: DO k=1, np
            p = pd(k)
            w = 2.*3.141592654/p
            !
            ! *****CHOICE OF INTERVAL OF INTEGRATION*****
            !
            delp = p/20.0
            ! L=DEL/DELP+1.-1.E-05
            ni = CEILING(del/delp)
            delt = del / REAL( ni )
            !
            ! *****COMPUTATION OF MATRICES A AND B*****
            !
            CALL calcAB(d, w, delt, A, B)
            !
            ! **********INITIATION**********
            !
            x(1) = 0.0
            x(3) = 0.0
            x(2) = 0.0
            dmax = 0.0
            vmax = 0.0
            amax = 0.0
            i = 1
            dw = 2.* w * d
            w2 = w * w
            IF (d < UNDAMPED) THEN
                ! for undamped system, more iterations are required to find the true
                ! maximum
                nadd = 2.*p / delt + 1.e-05
            ELSE
                nadd = 0
            END IF
            !
            ! *****COMPUTATION OF RESPONSE*****
            !
            ga2 = ga(1)   ! ga(1)

            LACC: DO i=1, na + nadd
                ! *****TEST FOR END OF INTEGRATION*****
                IF (i == na ) vend = SQRT (x(2)**2 + w2*(x(1)**2) )
                ga1 = ga2
                IF (i < na) THEN
                    ga2 = ga(i+1)
                ELSE
                    ga2 = 0.0  ! zero padding for undamped system
                END IF
!!$7               sl = (ga(i+1)-ga(i))/REAL(ni)
                sl = (ga2 - ga1) / ni
                LINTEGRATE: DO  m = 1, ni
                    g(1) = (ga1 + sl*(m-1)) * sf
                    g(2) = g(1) + sl * sf
                    ! g(2) = (ga(i) + sl*REAL(m))*sf
                    ty(1) = A(1,1)*x(1) + A(1,2)*x(2) + B(1,1)*g(1) + B(1,2)*g(2)
                    ty(2) = A(2,1)*x(1) + A(2,2)*x(2) + B(2,1)*g(1) + B(2,2)*g(2)
                    ty(3) = -(dw*ty(2) + w2*ty(1))
                    !
                    ! *****MONITORING THE MAX= VALUES*****
                    !
!!$                IF(ABS(TY(1)).LE.ABS(DMAX)) GO TO 14
!!$                DMAX=TY(1)
!!$14              X(1)=TY(1)
!!$                IF(ABS(TY(2)).LE.ABS(VMAX)) GO TO 15
!!$                VMAX=TY(2)
!!$15              X(2)=TY(2)
!!$                IF(ABS(TY(3)).LE.ABS(AMAX)) GO TO 16
!!$                AMAX=TY(3)
!!$16              X(3)=TY(3)
                    x(1) = ty(1)
                    x(2) = ty(2)
                    x(3) = ty(3)
                    absty = ABS(ty)
                    IF (absty(1) > dmax)  dmax = absty(1)
                    IF (absty(2) > vmax)  vmax = absty(2)
                    IF (absty(3) > amax)  then
                        amax = absty(3)
                        if (2*m >= ni) then
                            iamax = i + 1
                        else
                            iamax = i
                        endif
                        psign = sign(1.0, ty(3))
                        ! print*, psign, ty(3)
                    end if

                END DO LINTEGRATE  ! m for integration loop (within each time step)
            END DO LACC  ! i for time step

            ! J = 1, num_damping_ratios
            ! K = 1, num_periods
            IF (d < UNDAMPED) fs(k) = vend  ! for undamped oscilator, Fourier spectrum ?
            SD(J,K) = DMAX
            SV(J,K) = VMAX
            SA(J,K) = AMAX
            TA(J,K) = iamax * del ! use coarser del than delt [for intentration]
            PA(J,k) = psign
            ! print*, psign, PA(J, K)
!!$
!!$            IF(i == na) go TO 18
!!$            GO TO 19
!!$18          VEND = SQRT (X(2)**2 + W2*(X(1)**2) )
!!$19          IF(I.EQ.(NA+NADD)) GO TO 8
!!$            IF(I.GE.NA) GO TO 17
!!$            GO TO 7
!!$17          GA(I+1)=0.0
!!$            GO TO 7
!!$8           IF(D.GE.1.E-03) GO TO 26
!!$            FS(K) = VEND
!!$26          SD(J,K)=DMAX
!!$            SV(J,K)=VMAX
!!$            SA(J,K)=AMAX
        END DO LPERIOD  ! k for period loop
        !$omp end do nowait
    !$omp end parallel 
    END DO LDAMP  ! j for damping loop

END SUBROUTINE rst_exactmethod

! ------------------------------------------------------------------------------
SUBROUTINE calcAB(d, w, delt, A, B)
    ! the sign of B is opposite to that in Navin C. Nigam and Paul C. Jennings
    REAL, INTENT(in) :: d, w, delt
    REAL, INTENT(out), DIMENSION (2,2) :: A, B

    REAL dw, d2, a0, a1, ad1, a2, a3, w2, a4, a5, a6, a7, a8, a9, a10
    REAL a11, a12, a13, a14, a15, a16, a17

    dw = d*w
    d2 = d*d
    a0 = EXP(-dw * delt)
    a1 = w * SQRT(1. - d2)
    ad1 = a1 * delt
    a2 = SIN(ad1)
    a3 = COS(ad1)
    w2 = w*w
    a4 = (2.*d2 - 1.)/w2
    a5 = d/w
    a6 = 2.*a5/w2
    a7 = 1./w2
    a8 = (a1*a3 - dw*a2)*a0
    a9 =-(a1*a2 + dw*a3)*a0
    a10 = a8/a1
    a11 = a0/a1
    a12 = a11*a2
    a13 = a0*a3
    a14 = a10*a4
    a15 = a12*a4
    a16 = a6*a13
    a17 = a9*a6
    A(1,1) = a0*(dw*a2/a1 + a3)
    A(1,2) = a12
    A(2,1) = a10*dw + a9
    A(2,2) = a10
    B(1,1) =(a15 + a16 - a6)/delt + a12*a5 + a7*a13
    B(1,2) =(-a15 - a16 + a6)/delt - a7
    B(2,1) =(a14 + a17 + a7)/delt + a10*a5 + a9*a7
    B(2,2) =(-a14 - a17 - a7)/delt
END SUBROUTINE calcAB


! ------------------------------------------------------------------------------
! CARES frequencies at which response spectrum is calculated
SUBROUTINE fillfreq(dt, frq, nfrq)
    !f2py real, intent(in) :: dt
    !f2py integer, intent(hide), depend(dt) :: nfrq=50.0*log10(1.0/(2.0*dt*0.1*0.954993))
    !f2py real, dimension(nfrq), intent(out) :: frq

    REAL, INTENT(in) :: dt
    REAL, DIMENSION(nfrq), INTENT(out) :: frq
    integer :: nfrq

    ! REAL :: frqcut
    INTEGER :: i

    !~ frqcut = 1./(2.0 * dt)
    !~ nfrq = 50.0*log10(frqcut / (0.1*0.954993) )
    !~ allocate(frq(nfrq))

    forall (i = 1:nfrq) frq(i) = 0.10 * 0.954993 * (10.**(.020 * i))
    ! set special values
    if (nfrq >= 71) frq(71) = 2.5
    if (nfrq >= 86) frq(86) = 5.0  ! OLD 87
    if (nfrq >= 121) frq(121) = 25.0  ! OLD 123

END SUBROUTINE fillfreq

! ------------------------------------------------------------------------------

