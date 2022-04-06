// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail

// copied from https://github.com/hughperkins/EasyCL, apr 6, 2022

#include <string>
#include <vector>
#include <sstream>
using namespace std;

#include "stringhelper.h"

namespace easycl {
    vector<string> split(const string &str, const string &separator) {
	    vector<string> splitstring;
	    int start = 0;
	    int npos = (int)str.find(separator);
	    while (npos != (int)str.npos) {
		    splitstring.push_back(str.substr(start, npos-start));
		    start = npos + (int)separator.length();
		    npos = (int)str.find(separator, start);
	    }
	    splitstring.push_back(str.substr(start) );
        return splitstring;
    }

    string trim(const string &target) {

       int origlen = (int)target.size();
       int startpos = -1;
       for(int i = 0; i < origlen; i++) {
          if(target[i] != ' ' && target[i] != '\r' && target[i] != '\n') {
             startpos = i;
             break;
          }
       }
       int endpos = -1;
       for(int i = origlen - 1; i >= 0; i--) {
          if(target[i] != ' ' && target[i] != '\r' && target[i] != '\n') {
             endpos = i;
             break;
          }      
       }
       if(startpos == -1 || endpos == -1) {
          return "";
       }
       return target.substr(startpos, endpos-startpos + 1);
    }

    string replace(string targetString, string oldValue, string newValue) {
        size_t pos = targetString.find(oldValue);
        if(pos == string::npos) {
            return targetString;
        }
        return targetString.replace(pos, oldValue.length(), newValue);
    }
    string replaceGlobal(string targetString, string oldValue, string newValue) {
        int pos = 0;
        string resultString = "";
        size_t targetPos = targetString.find(oldValue, pos);
        while(targetPos != string::npos) {
            string preOld = targetString.substr(pos, targetPos - pos);
            resultString += preOld + newValue;
            pos = targetPos + oldValue.length();
            targetPos = targetString.find(oldValue, pos);
        }
        resultString += targetString.substr(pos);
        return resultString;
    }

    std::string toLower(std::string in) {
         int len = static_cast<int>(in.size());
         char *buffer = new char[len + 1];
         for(int i = 0; i < len; i++) {
            char thischar = in[i];
            thischar = tolower(thischar);
            buffer[i] = thischar;
        }
        buffer[len] = 0;
        std::string result = std::string(buffer);
        delete[] buffer;
        return result;
    }

    void strcpy_safe(char *destination, char const*source, int maxLength) {
        int i = 0;
        for(i = 0; i < maxLength; i++) {
            destination[i] = source[i];
            if(source[i] == 0) {
                break;
            }
        }
        destination[i] = 0;
    }
}

