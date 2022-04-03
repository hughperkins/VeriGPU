#include <iostream>
#include <bitset>
#include <iomanip>


void test_float(float fval) {
    unsigned int ival = reinterpret_cast<unsigned int &>(fval);
    std::bitset<32> bits(ival);
    float new_f = reinterpret_cast<float &>(ival);
    std::cout << fval << " " << new_f << " " << ival << " " << std::hex << ival << " " << std::dec << bits << std::endl;
}


int main(int argc, char *argv[]) {
    test_float(1.2);
    test_float(3.5);
    test_float(0.123);
    test_float(-2.5);
    test_float(1.0);
    test_float(2.0);
    test_float(4.0);
    test_float(4.2);
    test_float(4.5);
    return 0;
}
