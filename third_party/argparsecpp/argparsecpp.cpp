// Copyright Hugh Perkins 2016

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "argparsecpp.h"

#include <string>
#include <map>
#include <set>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <iostream>


namespace argparsecpp {

OptionBool::OptionBool(bool *target) : target(target) {}
bool OptionBool::needsValue() { return false; }
OptionBool *OptionBool::defaultTrue() {
    this->_defaultTrue = true;
    return this;
}
void OptionBool::parse(std::string valueString) {
    *target = !_defaultTrue;
}
void OptionBool::applyDefault() {
    *target = _defaultTrue;
}
void OptionBool::writeDefault(std::ostream &os) {
    os << _defaultTrue;
}

OptionString::OptionString(std::string *target) : target(target) {}
bool OptionString::needsValue() { return true; }
void OptionString::parse(std::string valueString) {
    *target = valueString;
}
OptionString *OptionString::defaultValue(std::string _default) {
    this->_default = _default;
    return this;
}
void OptionString::applyDefault() {
    *target = _default;
}
void OptionString::writeDefault(std::ostream &os) {
    os << _default;
}

OptionFloat::OptionFloat(float *target) : target(target) {}
bool OptionFloat::needsValue() { return true; }
void OptionFloat::parse(std::string valueString) {
    *target = atof(valueString.c_str());
}
OptionFloat *OptionFloat::defaultValue(float _default) {
    this->_default = _default;
    return this;
}
void OptionFloat::applyDefault() {
    *target = _default;
}
void OptionFloat::writeDefault(std::ostream &os) {
    os << _default;
}

OptionInt::OptionInt(int *target) : target(target) {}
bool OptionInt::needsValue() { return true; }
void OptionInt::parse(std::string valueString) {
    *target = atoi(valueString.c_str());
}
OptionInt *OptionInt::defaultValue(int _default) {
    this->_default = _default;
    return this;
}
void OptionInt::applyDefault() {
    *target = _default;
}
void OptionInt::writeDefault(std::ostream &os) {
    os << _default;
}

OptionInt *ArgumentParser::add_int_argument(std::string option, int *var) {
    std::unique_ptr<OptionInt> option_(new OptionInt(var));
    OptionInt *res = option_.get();
    options[option] = std::move(option_);
    return res;
}
OptionBool *ArgumentParser::add_bool_argument(std::string option, bool *var) {
    std::unique_ptr<OptionBool> option_(new OptionBool(var));
    OptionBool *res = option_.get();
    options[option] = std::move(option_);
    return res;
}
OptionFloat *ArgumentParser::add_float_argument(std::string option, float *var) {
    std::unique_ptr<OptionFloat> option_(new OptionFloat(var));
    OptionFloat *res = option_.get();
    options[option] = std::move(option_);
    return res;
}
OptionString *ArgumentParser::add_string_argument(std::string option, std::string *var) {
    std::unique_ptr<OptionString> option_(new OptionString(var));
    OptionString *res = option_.get();
    options[option] = std::move(option_);
    return res;
}
void ArgumentParser::print_usage() {
    std::cout << std::endl;
    std::cout << "Usage:" << std::endl;
    for(auto it=options.begin(); it != options.end(); it++) {
        Option *option = it->second.get();
        std::string arg = it->first;
        std::cout << " " << arg;
        if(option->_help != "") {
            std::cout << "     " + option->_help;
        }
        if(option->_required) {
            std::cout << " (required)";
        } else {
            std::cout << " [default: ";
            option->writeDefault(std::cout);
            std::cout << "]";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
bool ArgumentParser::parse_args(int argc, char *argv[]) {
    std::set<std::string> seenOptions;
    for(auto it=options.begin(); it != options.end(); it++) {
        Option *option = it->second.get();
        option->applyDefault();
    }
    for(int i = 1; i < argc; i++) {
        std::string thisArg = argv[i];
        if(thisArg == "-h" || thisArg == "-?" || thisArg == "--help") {
            print_usage();
            return false;
        }
        if(options.find(thisArg) == options.end()) {
            print_usage();
            std::cout << "unknown option " << thisArg << std::endl;
            return false;
        }
        seenOptions.insert(thisArg);
        Option *option = options[thisArg].get();
        if(option->needsValue()) {
            if(i + 1 >= argc) {
                print_usage();
                std::cout << "no value found for optoin " << thisArg << std::endl;
                return false;
            }
            std::string valueString = argv[i + 1];
            option->parse(valueString);
            i++;
        } else {
            option->parse("");
        }
    }
    // check for required options
    for(auto it=options.begin(); it != options.end(); it++) {
        Option *option = it->second.get();
        std::string arg = it->first;
        if(option->_required) {
            if(seenOptions.find(arg) == seenOptions.end()) {
                print_usage();
                std::cout << "Please provide a value for " + arg << std::endl;
                std::cout << std::endl;
                return false;
            }
        }
    }
    return true;
}

} // argparsecpp
