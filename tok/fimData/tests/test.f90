module math_mod
contains
    function add(a, b) result(c)
        integer :: a, b, c
        c = a + b
    end function add
end module math_mod

program main
    print *, 'hi'
end program main

subroutine square(n)
    integer :: n
    n = n * n
end subroutine square
