-module(math).
-export([add/2]).

add(A, B) ->
    A + B.

square(N) ->
    N * N.

-record(point, {x, y}).
