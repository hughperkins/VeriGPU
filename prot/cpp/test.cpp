void out(int val);

int foo(int one, int two) {
    out(one);
    out(two);
    return one * two;
}

void foo2(int start, int end, int*output) {
    int sum = 0;
    for(int i=start; i < end; i++) {
        sum += i;
        out(sum);
    }
    *output = sum;
}
