#include <iostream>
#include <fstream>
#include <bitset>

uint32_t upper(uint32_t val) {
    uint32_t bit11 = ((val >> 11) % 2);
    std::cout << "bit11=" << bit11 << std::endl;
    return (val >> 12) + bit11;
}

uint32_t lower(uint32_t val) {
    // the return number will be signeed, in terms of bits, just stored in unsigned type
    return val % (1 << 12);
}

int main(int argc, char *argv[]) {
    std::cout << "argv[1]" << argv[1] << std::endl;
    uint32_t val = atoi(argv[1]);
    std::cout << "val " << val << std::endl;
    uint32_t u = upper(val);
    uint32_t l = lower(val);
    std::cout << "upper " << u << std::endl;
    std::cout << "lower " << l << std::endl;

    std::bitset<32> vb(val);
    std::bitset<32> ub(u);
    std::bitset<32> lb(l);

    std::cout << "val   " << vb << std::endl;
    std::cout << "upper " << ub << std::endl;
    std::cout << "lower " << lb << std::endl;

    int32_t sl = l;
    if(sl > 2047) {
        sl -= 4096;
    }
    std::cout << "sl " << sl << std::endl;
    uint32_t reconstr = (u << 12) + sl;
    std::cout << "reconstr " << reconstr << std::endl;

    return 0;
}
