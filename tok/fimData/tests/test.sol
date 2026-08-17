pragma solidity ^0.8.0;

contract Counter {
    uint256 private count;

    function increment() public {
        count += 1;
    }
}

interface IShape {
    function area() external view returns (uint256);
}

library MathLib {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
}
