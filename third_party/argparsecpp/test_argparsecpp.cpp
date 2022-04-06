#include "argparsecpp.h"

#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
    string requiredstring;
    bool somebool;
    bool bool_defaultrue;
    string somestring;
    string somestringwithdefault;
    int someint;
    float somefloat;
    argparsecpp::ArgumentParser parser;
    parser.add_string_argument("--requiredstring", &requiredstring)->required();
    parser.add_string_argument("--somestringwithdefault", &somestringwithdefault)->defaultValue("foobar");
    parser.add_bool_argument("--somebool", &somebool);
    parser.add_bool_argument("--no-bool_defaulttrue", &bool_defaultrue)->defaultTrue();
    parser.add_string_argument("--somestring", &somestring)->help("this should be a string");
    parser.add_int_argument("--someint", &someint);
    parser.add_float_argument("--somefloat", &somefloat);
    if(!parser.parse_args(argc, argv)) {
        return -1;
    }
    cout << "requiredstring " << requiredstring << endl;
    cout << "somestringwithdefault " << somestringwithdefault << endl;
    cout << "somebool " << somebool << endl;
    cout << "bool_defaultrue " << bool_defaultrue << endl;
    cout << "somestring " << somestring << endl;
    cout << "someint " << someint << endl;
    cout << "somefloat " << somefloat << endl;
    return 0;
}
