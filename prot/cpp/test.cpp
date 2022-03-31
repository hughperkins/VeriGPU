void out(int val);

void call_with_ptr(int strt, int end, int *output);

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

void foo3(int start, int end, int *output) {
    call_with_ptr(start, end, output);
}
