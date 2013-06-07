      subroutine asubprod(aprod, ix, ntrue, nfree, wx, wy, x, y)

      implicit none
      external aprod
      integer ntrue, nfree
      integer ix(nfree)
      double precision wx(ntrue), wy(ntrue), x(nfree), y(nfree)

c     ------------------------------------------------------------------
c     Compute the product y = A*x, where A is symmetric.
c     ------------------------------------------------------------------

      double precision zero
      parameter(zero=0.0d0)
      integer i

c     Scatter.
      do i = 1, ntrue
         wx(i) = zero
      end do
      do i = 1, nfree
         wx(ix(i)) = x(i)
      end do

c     wy = A*wx
      call aprod(ntrue, wx, wy)

c     Gather.
      do i = 1, nfree
         y(i) = wy(ix(i))
      end do
      
      end
