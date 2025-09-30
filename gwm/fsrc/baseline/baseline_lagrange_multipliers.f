C
C     Subroutine to perform baseline correction using Lagrange multipliers.
C     to ensure zero final Vel, Displ, and average displ.
C     Initial conditions: a(0)=V(0)=d(0)=0.0
C      subroutine corr(n,a,c1,dt)
      subroutine baseline_lagrange_multipliers(n, a, dt)
      real a(n),dt
Cf2py intent(in,out,copy) a
Cf2py intent(in) dt
Cf2py integer intent(hide),depend(a) :: n=len(a)      
      real c1(n,3)
      real c(3,3),x(3),y(3),t1
C a(n) is the acc time history to be baseline corrected 
      nn=n-1
C determine alpha, beta and gama in appendix as c1, c2 and c3
      do 1 i=1,nn
      c1(i,1)=dt
      c1(i,2)=dt*dt*(n-i)
  1   continue
      c1(n,1)=dt/2.0e00
      c1(n,2)=dt*dt/4.0e00
      do 3 i=1,n
      t1=0.0e00
      do 2 i1=i,n
      t1=t1+c1(i1,2)
 2    continue
      c1(i,3)=c1(i,2)*dt/2.0e00+dt*t1
 3    continue
      do 4 i=1,3
      y(i)=0.0e00
      do 4 j=1,3
 4    c(i,j)=0.0e00
C Assemble coefs in Eq (10) for determining the Lagrange multipliers: [C]{x} = {y}
      do 5 i=1,n
      c(1,1)=c(1,1)+c1(i,1)*c1(i,1)
      c(2,2)=c(2,2)+c1(i,2)*c1(i,2)
      c(3,3)=c(3,3)+c1(i,3)*c1(i,3)
      c(1,2)=c(1,2)+c1(i,1)*c1(i,2)
      c(1,3)=c(1,3)+c1(i,1)*c1(i,3)
      c(2,3)=c(2,3)+c1(i,2)*c1(i,3)
      y(1)=y(1)+c1(i,1)*a(i)
      y(2)=y(2)+c1(i,2)*a(i)
      y(3)=y(3)+c1(i,3)*a(i)
  5   continue
      c(2,1)=c(1,2)
      c(3,1)=c(1,3)
      c(3,2)=c(2,3)
      n1=3
C Solve for the Lagrange multipliers
      call solve(c,y,n1,x)
C Baseline correct the acc time history a(n) using eq. (9)
      do 6 i=1,n
      a(i)=a(i)-x(1)*c1(i,1)-x(2)*c1(i,2)-x(3)*c1(i,3)
  6   continue
      return
      end
c  
c   solve linear equations C*X = Y.
c
      SUBROUTINE  SOLVE(A,B,N,X)
      real A(3,3),B(3),X(3),L,L1,L2,L3
      DO 1 I=2,N
 1    A(1,I)=A(1,I)/A(1,1)
      DO 2 I=2,N
      DO 3 J=2,I
      J2=J-1
      L=0.0
      DO 4 K=1,J2
 4    L=L+A(I,K)*A(K,J)
 3    A(I,J)=A(I,J)-L
      IF(I.GT.N-1) GO TO 2
      I3=I+1
      DO 5 J1=I3,N
      I1=I-1
      L1=0.0
      DO 6 K1=1,I1
 6    L1=L1+A(I,K1)*A(K1,J1)
 5    A(I,J1)=(A(I,J1)-L1)/A(I,I)
 2    CONTINUE
C  FORWARD SUBSTITUTION
C
      X(1)=B(1)/A(1,1)
      DO 7 I=2,N
      II=I-1
      L2=0.0
      DO 8 K=1,II
  8   L2=L2+X(K)*A(I,K)
  7   X(I)=(B(I)-L2)/A(I,I)
C  BACKWARD SUBSTITUTION
C   
      N2=N-1
      DO 9 I=1,N2
      L3=0.0
      DO 11 K=1,I
  11  L3=L3+X(N-K+1)*A(N-I,N-K+1)
   9  X(N-I)=X(N-I)-L3
      RETURN
      END
C
C  COMPUTE VELOCITY AND DISPLACEMENT TIME HISTORIES
C  FOR GIVEN ACCELEROGRAM	     
      subroutine avd(a,v,d,nrcds,dt)
      dimension a(nrcds),v(nrcds),d(nrcds)
Cf2py intent(in) a, dt
Cf2py integer intent(hide),depend(a) :: nrcds=len(a)
Cf2py intent(out) v, d
      v(1)=a(1)*dt/2.0
      d(1)=a(1)*dt*dt/6.0
      nn=nrcds-1
      vel=v(1)
      do i=1,nn
      vel=vel+(a(i+1)+a(i))*dt/2.0
      v(i+1)=vel
      d(i+1)=d(i)+v(i)*dt+(a(i+1)/6.0+a(i)/3.0)*dt*dt
      end do
      return 
      end


C
C COMPUTE COVs
C
      subroutine covs(a1,a2,cov12,nrcds,dt)
      dimension a1(nrcds),a2(nrcds)
Cf2py intent(in) a1, a2, dt
Cf2py intent(out) cov12
Cf2py integer intent(hide),depend(a) :: nrcds=len(a1)
      tt=nrcds*dt
      td=a1(1)*dt/2.0
      nn=nrcds-1
      vel1=td
      vel12=a1(1)*a1(1)*dt/3.0
      do i=1,nn
      vel1=vel1+(a1(i+1)+a1(i))*dt/2.0
      vel12=vel12+(a1(i+1)*a1(i)+(a1(i+1)-a1(i))**2/3.0)*dt
      end do
      vel1=vel1/tt
      vel12=vel12/tt
      sig1=sqrt(vel12-vel1*vel1)
      td=a2(1)*dt/2.0
      vel2=td
      vel22=a2(1)*a2(1)*dt/3.0
      do i=1,nn
      vel2=vel2+(a2(i+1)+a2(i))*dt/2.0
      vel22=vel22+(a2(i+1)*a2(i)+(a2(i+1)-a2(i))**2/3.0)*dt
      end do
      vel2=vel2/tt
      vel22=vel22/tt
      sig2=sqrt(vel22-vel2*vel2)
      var12=a1(1)*a2(1)*dt/3.0
      do i=1,nn
      da1=a1(i+1)-a1(i)
      da2=a2(i+1)-a2(i)
      var12=var12+(a1(i)*a2(i)+(a2(i)*da1+a1(i)*da2)/2.0
     &+da1*da2/3.0)*dt
      end do
      var12=var12/tt-vel1*vel2
      cov12=var12/sig1/sig2
      return 
      end