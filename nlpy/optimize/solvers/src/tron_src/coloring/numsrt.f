      subroutine numsrt(n,nmax,num,mode,index,last,next)
      integer n,nmax,mode
      integer num(n),index(n),last(0:nmax),next(n)
c     **********.
c
c     subroutine numsrt
c
c     Given a sequence of integers, this subroutine groups
c     together those indices with the same sequence value
c     and, optionally, sorts the sequence into either
c     ascending or descending order.
c
c     The sequence of integers is defined by the array num,
c     and it is assumed that the integers are each from the set
c     0,1,...,nmax. On output the indices k such that num(k) = l
c     for any l = 0,1,...,nmax can be obtained from the arrays
c     last and next as follows.
c
c           k = last(l)
c           while (k .ne. 0) k = next(k)
c
c     Optionally, the subroutine produces an array index so that
c     the sequence num(index(i)), i = 1,2,...,n is sorted.
c
c     The subroutine statement is
c
c       subroutine numsrt(n,nmax,num,mode,index,last,next)
c
c     where
c
c       n is a positive integer input variable.
c
c       nmax is a positive integer input variable.
c
c       num is an input array of length n which contains the
c         sequence of integers to be grouped and sorted. It
c         is assumed that the integers are each from the set
c         0,1,...,nmax.
c
c       mode is an integer input variable. The sequence num is
c         sorted in ascending order if mode is positive and in
c         descending order if mode is negative. If mode is 0,
c         no sorting is done.
c
c       index is an integer output array of length n set so
c         that the sequence
c
c               num(index(i)), i = 1,2,...,n
c
c         is sorted according to the setting of mode. If mode
c         is 0, index is not referenced.
c
c       last is an integer output array of length nmax + 1. The
c         index of num for the last occurrence of l is last(l)
c         for any l = 0,1,...,nmax unless last(l) = 0. In
c         this case l does not appear in num.
c
c       next is an integer output array of length n. If
c         num(k) = l, then the index of num for the previous
c         occurrence of l is next(k) for any l = 0,1,...,nmax
c         unless next(k) = 0. In this case there is no previous
c         occurrence of l in num.
c
c     Argonne National Laboratory. MINPACK Project. July 1983.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer i,j,jinc,jl,ju,k,l
c
c     Determine the arrays next and last.
c
      do 10 i = 0, nmax
         last(i) = 0
   10    continue
      do 20 k = 1, n
         l = num(k)
         next(k) = last(l)
         last(l) = k
   20    continue
      if (mode .eq. 0) return
c
c     Store the pointers to the sorted array in index.
c
      i = 1
      if (mode .gt. 0) then
         jl = 0
         ju = nmax
         jinc = 1
      else
         jl = nmax
         ju = 0
         jinc = -1
         end if
      do 50 j = jl, ju, jinc
         k = last(j)
   30    continue
            if (k .eq. 0) go to 40
            index(i) = k
            i = i + 1
            k = next(k)
            go to 30
   40    continue
   50    continue
      return
c
c     Last card of subroutine numsrt.
c
      end
