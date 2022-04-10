void out(unsigned int val);

void prime_numbers_cpp() {
    for (unsigned int candidate = 2; candidate < 32; candidate++) {
        unsigned int is_prime = 1;
        for (unsigned int test = 2; test < candidate; test++)
        {
            if (candidate % test == 0) {
                is_prime = 0;
                break;
            }
        }
        if(is_prime) {
            out(candidate);
        }
    }
}
