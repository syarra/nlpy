      subroutine dgl2sp(nx,ny,nnz,indrow,indcol)
      integer nx, ny, nnz
      integer indrow(*), indcol(*)
c     **********
c
c     Subroutine dgl2sp
c
c     This subroutine defines the sparsity structure of the Hessian
c     matrix for the Ginzburg-Landau (2-dimensional) problem.
c
c     The subroutine statement is
c
c       subroutine dgl2sp(nx,ny,nnz,indrow,indcol)
c
c     where
c
c       nx is an integer variable.
c         On entry nx is the number of grid points in the first
c            coordinate direction.
c         On exit nx is unchanged.
c
c       ny is an integer variable.
c         On entry ny is the number of grid points in the second
c            coordinate direction.
c         On exit ny is unchanged.
c
c       nnz is an integer variable.
c         On entry nnz need not be specified.
c         On exit nnz is set to the number of nonzeros in the
c            lower triangle of the Hessian matrix.
c
c       indrow is an integer array of dimension at least nnz.
c         On entry indrow need not be specified.
c         On exit indrow contains the row indices of the nonzeros
c            in the lower triangle of the Hessian matrix.
c
c       indcol is an integer array of dimension at least nnz.
c         On entry indcol need not be specified.
c         On exit indcol contains the column indices of the nonzeros
c            in the lower triangle of the Hessian matrix.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Richard Carter, Paul Plassmann, and Steve Wright.
c
c     **********
      integer dof, i, ix, iy, j, k, n

c     Compute the diagonal

      n = 4*nx*ny
      do 10 i = 1, n
         indrow(i) = i
         indcol(i) = i
   10 continue

c     Order each of the four nodal degrees of freedom sequentially for
c     each vertex (ix,iy), ix = 1, ..., nx , iy = 1, ..., ny.

      nnz = n
      j = 0
      do 50 iy = 1, ny
         do 40 ix = 1, nx
            do 30 dof = 1, 4
               j = j + 1
               do 20 k = dof + 1, 4
                  nnz = nnz + 1
                  indrow(nnz) = 4*(nx*(iy-1)+ix-1) + k
                  indcol(nnz) = j
   20          continue

c              East vertex.

               if (ix .ne. nx) then
                  k = 4*(nx*(iy-1)+ix) + 1
                  if (dof .le. 3) then
                     nnz = nnz + 2
                     indrow(nnz-1) = k
                     indcol(nnz-1) = j
                     indrow(nnz) = k + 1
                     indcol(nnz) = j
                     if (dof .eq. 3) then
                        nnz = nnz + 1
                        indrow(nnz) = k + 3
                        indcol(nnz) = j
                     end if
                  else
                     nnz = nnz + 1
                     indrow(nnz) = k + 3
                     indcol(nnz) = j
                  end if
               end if

c              West column of grid.

               if (ix .eq. 1) then
                  k = 4*(nx*(iy-1)+nx-1) + 1
                  if (dof .le. 2) then
                     nnz = nnz + 3
                     indrow(nnz-2) = k
                     indcol(nnz-2) = j
                     indrow(nnz-1) = k + 1
                     indcol(nnz-1) = j
                     indrow(nnz) = k + 2
                     indcol(nnz) = j
                  else if (dof .eq. 4) then
                     nnz = nnz + 2
                     indrow(nnz-1) = k + 2
                     indcol(nnz-1) = j
                     indrow(nnz) = k + 3
                     indcol(nnz) = j
                  end if
               end if

c              North-west vertex (if not on western column).

               if ((iy .ne. ny) .and. (ix .ne. 1)) then
                  k = 4*(nx*iy+ix-2) + 1
                  if (dof .eq. 4) then
                     nnz = nnz + 1
                     indrow(nnz) = k + 2
                     indcol(nnz) = j
                  end if
               end if

c              North vertex.

               if (iy .ne. ny) then
                  k = 4*(nx*iy+ix-1) + 1
                  if (dof .le. 2) then
                     nnz = nnz + 2
                     indrow(nnz-1) = k
                     indcol(nnz-1) = j
                     indrow(nnz) = k + 1
                     indcol(nnz) = j
                  else if (dof .eq. 3) then
                     nnz = nnz + 1
                     indrow(nnz) = k + 2
                     indcol(nnz) = j
                  else
                     nnz = nnz + 3
                     indrow(nnz-2) = k
                     indcol(nnz-2) = j
                     indrow(nnz-1) = k + 1
                     indcol(nnz-1) = j
                     indrow(nnz) = k + 2
                     indcol(nnz) = j
                  end if
               end if

c              North-west vertex (if on western column).

               if ((iy .ne. ny) .and. (ix .eq. 1)) then
                  k = 4*(nx*iy+nx-1) + 1
                  if (dof .eq. 4) then
                     nnz = nnz + 1
                     indrow(nnz) = k + 2
                     indcol(nnz) = j
                  end if
               end if

c              South-east vertex (if on south-east corner).

               if ((iy .eq. 1) .and. (ix .eq. nx)) then
                  k = 4*nx*(ny-1) + 1
                  if (dof .eq. 3) then
                     nnz = nnz + 1
                     indrow(nnz) = k + 3
                     indcol(nnz) = j
                  end if
               end if

c              South row of grid.

               if (iy .eq. 1) then
                  k = 4*(nx*(ny-1)+ix-1) + 1
                  if (dof .le. 2) then
                     nnz = nnz + 3
                     indrow(nnz-2) = k
                     indcol(nnz-2) = j
                     indrow(nnz-1) = k + 1
                     indcol(nnz-1) = j
                     indrow(nnz) = k + 3
                     indcol(nnz) = j
                  else if (dof .eq. 3) then
                     nnz = nnz + 2
                     indrow(nnz-1) = k + 2
                     indcol(nnz-1) = j
                     indrow(nnz) = k + 3
                     indcol(nnz) = j
                  end if
               end if

c              South-east vertex (if on south border, but not on
c              south-east corner).

               if ((iy .eq. 1) .and. (ix .ne. nx)) then
                  k = 4*(nx*(ny-1)+ix) + 1
                  if (dof .eq. 3) then
                     nnz = nnz + 1
                     indrow(nnz) = k + 3
                     indcol(nnz) = j
                  end if
               end if

   30       continue
   40    continue
   50 continue

c     Reorder by degree of freedom.

      do 60 k = 1, nnz
         j = mod(indrow(k),4) - 1
         if (j .eq. -1) j = 3
         indrow(k) = j*nx*ny + (indrow(k)-1)/4 + 1
         j = mod(indcol(k),4) - 1
         if (j .eq. -1) j = 3
         indcol(k) = j*nx*ny + (indcol(k)-1)/4 + 1
   60 continue

c     Make sure all elements are in the lower half.

      do 70 k = 1, nnz
         if (indcol(k) .gt. indrow(k)) then
            i = indrow(k)
            indrow(k) = indcol(k)
            indcol(k) = i
         end if
   70 continue

      end
