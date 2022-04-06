# ArgParseCpp
C++ version of Python's ArgParse

Loosely modelled off Python's argparse module.

I couldnt find a drop-in portable commandline parser for c++.  Options that came up on stackoverflow:
- use boost
- use getopt

Boost is hard to compile on Windows, so I'd prefer to avoid linking with thta. Especially for something as simple as parsing a commandline.

Getopt seems to have some combination of not being portable, being GPL-ish, and not accepting long arguments.

By comparison, ArgParseCpp:
- Apache license.  Simply drop the files into your own repo.  dont modify the copyright headers on them.  Ideally star my repo :-)
- Doesnt need any dependencies.  Well, it's sort of c++11.  Thats it
- Is totally portable

Example usage:

```
    string requiredstring = "";
    bool somebool = false;
    string somestring = "";
    int someint = 0;
    float somefloat = 0.0f;

    cocl::ArgumentParser parser;
    parser.add_string_argument("--requiredstring", &requiredstring)->required();
    parser.add_bool_argument("--somebool", &somebool);
    parser.add_string_argument("--somestring", &somestring)->help("this should be a string");
    parser.add_int_argument("--someint", &someint);
    parser.add_float_argument("--somefloat", &somefloat);

    if(!parser.parse_args(argc, argv)) {
        return -1;
    }

    cout << "requiredstring " << requiredstring << endl;
    cout << "somebool " << somebool << endl;
    cout << "somestring " << somestring << endl;
    cout << "someint " << someint << endl;
    cout << "somefloat " << somefloat << endl;
```

## Build

I reckon you can just drop the files directly into your own project.  So, no need to build

## Test

Run:
```
mkdir -p build
cd build
cmake ..
make
./test_argparsecpp
```
