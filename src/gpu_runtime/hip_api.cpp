#include <iostream>

extern "C" {
    void roctxMarkA() {
        std::cout << "roctxMarkA" << std::endl;
    }
    void roctxRangePop() {
        std::cout << "roctxRangePop" << std::endl;
    }
    void roctxRangePushA()
    {
        std::cout << "roctxRangePushA" << std::endl;
    }
    }
