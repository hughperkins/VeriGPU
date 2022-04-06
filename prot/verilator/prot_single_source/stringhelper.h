// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail

// copied from https://github.com/hughperkins/EasyCL, apr 6, 2022

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>

namespace VeriGPU {
   class IHasToString {
    public:
        virtual ~IHasToString() {}
        virtual std::string toString() = 0;
    };

    //std::string toString(IHasToString *val); // { // not terribly efficient, but works...
    //   std::ostringstream myostringstream;
    //   myostringstream << val->toString();
    //   return myostringstream.str();
    //}

    template<typename T>
    std::string toString(T val) { // not terribly efficient, but works...
       std::ostringstream myostringstream;
       myostringstream << val;
       return myostringstream.str();
    }

    std::vector<std::string> split(const std::string &str, const std::string &separator = " ");
    std::string trim(const std::string &target);

    inline float atof(std::string stringvalue) {
       return (float)std::atof(stringvalue.c_str());
    }
    inline int atoi(std::string stringvalue) {
       return std::atoi(stringvalue.c_str());
    }

    // returns empty string if off the end of the number of available tokens
    inline std::string getToken(std::string targetstring, int tokenIndexFromZero, std::string separator = " ") {
       std::vector<std::string> splitstring = split(targetstring, separator);
       if(tokenIndexFromZero < (int)splitstring.size()) {
          return splitstring[tokenIndexFromZero];
       } else {
          return "";
       }
    }

    std::string replace(std::string targetString, std::string oldValue, std::string newValue);
    std::string replaceGlobal(std::string targetString, std::string oldValue, std::string newValue);

    std::string toLower(std::string in);

    void strcpy_safe(char *destination, char const*source, int maxLength);
}
