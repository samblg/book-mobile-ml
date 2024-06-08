#ifndef MD5_H
#define MD5_H

struct MD5_CTX
{
    unsigned int count[2];
    unsigned int state[4];
    unsigned char buffer[64];
};

#include <string>
#include <istream>

class Md5Context {
public:
    Md5Context();
    ~Md5Context();
    void reset();
    void addData(const char* data, unsigned int size);
    void addData(std::istream& dataStream);
    std::string result();
    std::string hexResult();

private:
    MD5_CTX _context;
};
 
#endif
