module adder(input a, input b, output c);
    assign c = a + b;
endmodule

function integer add;
    input a;
    input b;
    add = a + b;
endfunction

task do_stuff;
    input x;
endtask
