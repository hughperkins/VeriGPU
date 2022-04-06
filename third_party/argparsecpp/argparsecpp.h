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

// loosely modeled off Python's `argparse`

#pragma once

#include <string>
#include <map>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <iostream>


namespace argparsecpp {

class Option {
public:
    virtual ~Option() {}
    virtual  bool needsValue() = 0;
    virtual void parse(std::string valueAsString = "") = 0;
    Option *required() {
        this->_required = true;
        return this;
    }
    Option *help(std::string help) {
        this->_help = help;
        return this;
    }
    bool _required = false;
    std::string _help = "";
    virtual void applyDefault() {};
    virtual void writeDefault(std::ostream &os) = 0;
};

class OptionBool : public Option {
public:
    OptionBool(bool *target);
    bool *target;
    bool _defaultTrue = false;
    virtual bool needsValue() override;
    virtual void parse(std::string valueString) override;
    OptionBool *defaultTrue();
    virtual void applyDefault() override;
    virtual void writeDefault(std::ostream &os) override;
};

class OptionString : public Option {
public:
    std::string _default = "";
    OptionString(std::string *target);
    std::string *target;
    virtual  bool needsValue() override;
    void parse(std::string valueString) override;
    OptionString *defaultValue(std::string _default);
    virtual void applyDefault() override;
    virtual void writeDefault(std::ostream &os) override;
};

class OptionFloat : public Option {
public:
    float _default = 0.0f;
    OptionFloat(float *target);
    float *target;
    virtual  bool needsValue() override;
    void parse(std::string valueString) override;
    OptionFloat *defaultValue(float _default);
    virtual void applyDefault() override;
    virtual void writeDefault(std::ostream &os) override;
};

class OptionInt : public Option {
public:
    int _default = 0;
    OptionInt(int *target);
    int *target;
    virtual  bool needsValue() override;
    void parse(std::string valueString) override;
    OptionInt *defaultValue(int _default);
    virtual void applyDefault() override;
    virtual void writeDefault(std::ostream &os) override;
};

class ArgumentParser {
public:
    std::map<std::string, std::unique_ptr<Option> > options;
    OptionInt *add_int_argument(std::string option, int *var);
    OptionBool *add_bool_argument(std::string option, bool *var);
    OptionFloat *add_float_argument(std::string option, float *var);
    OptionString *add_string_argument(std::string option, std::string *var);
    bool parse_args(int argc, char *argv[]);
    void print_usage();
};

} // namespace argparsecpp
