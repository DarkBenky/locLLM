module adder(input logic a, input logic b, output logic c);
    assign c = a + b;
endmodule

class Foo;
    int x;
endclass

interface bus_if(input logic clk);
endinterface

package pkg;
    function int add(int a, int b);
        return a + b;
    endfunction
endpackage
