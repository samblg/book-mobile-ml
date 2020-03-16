#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <set>

//01056332088

#ifndef WIN32
#define KLOG_PREDICT_BRANCH_NOT_TAKEN(x) (__builtin_expect(x, 0))
#else
#define KLOG_PREDICT_BRANCH_NOT_TAKEN(x) x
#endif

#ifndef DISABLE_LOG
#define LOG(severity) (dataee::logging::Logger( \
        (severity), \
        dataee::logging::LoggerStreamManager::GetInstance().getDefaultOutputStreams(severity), \
        __FILE__, \
        __LINE__ \
    ).reference())
#else
#define LOG(severity) !(KLOG_PREDICT_BRANCH_NOT_TAKEN(0)) ? \
    (void) 0 : \
    dataee::logging::LogMessageVoidify() & (dataee::logging::Logger( \
        (severity), \
        dataee::logging::LoggerStreamManager::GetInstance().getDefaultOutputStreams(severity), \
        __FILE__, \
        __LINE__ \
    ).reference())
#endif

#define LOG_IF(severity, expr) (dataee::logging::ExpressionLogger( \
        (severity), \
        dataee::logging::LoggerStreamManager::GetInstance().getDefaultOutputStreams((severity)), \
        __FILE__, \
        __LINE__, \
        expr, \
        #expr \
    ).reference())

#define LOG_CHECK(expr) !(KLOG_PREDICT_BRANCH_NOT_TAKEN(!(expr))) ? \
    (void) 0 : \
    dataee::logging::LogMessageVoidify() & dataee::logging::CheckLogger( \
        KLOG_FATAL, \
		dataee::logging::LoggerStreamManager::GetInstance().getDefaultOutputStreams(KLOG_FATAL), \
        __FILE__, \
        __LINE__, \
		!(KLOG_PREDICT_BRANCH_NOT_TAKEN(!(expr))), \
        #expr \
    ).reference()

#ifndef NDEBUG
#define DLOG(severity)
#define DLOG_IF(severity, expr)
#define DLOG_CHECK(expr)
#else
#define DLOG(severity) LOG(severity)
#define DLOG_IF(severity, expr) LOG_IF(severity, expr)
#define DLOG_CHECK(expr) LOG_CHECK(expr)
#endif //NDEBUG

#include "common/klog/check.h"

namespace dataee {
namespace logging {

class Logger;

}
}

namespace std {

dataee::logging::Logger& endl(dataee::logging::Logger& logger);

}

namespace dataee {
namespace logging {

enum class Severity {
    Debug,
    Info,
    Warning,
    Error,
    Fatal
};

class Logger {
public:
    Logger(Severity severity, std::ostream* outputStream,
           const std::string& sourceFile, int sourceLine, bool enableOutput = true);
    Logger(Severity severity, std::vector<std::ostream*> outputStreams,
           const std::string& sourceFile, int sourceLine, bool enableOutput = true);
    virtual ~Logger();

    template <class Value>
    Logger& output(Value value) {
        if ( !_enableOutput ) {
            return *this;
        }

        if ( !_inOutput ) {
            _inOutput = true;
            outputPrefix();
        }

        for ( std::ostream* outputStream : _outputStreams ) {
            *outputStream << value;
        }

        return *this;
    }

    void endl();

    template <class Value>
    Logger& operator<<(Value value) {
        return output(value);
    }

    Logger& operator<<(Logger& (*__pf)(Logger&))
    {
        return __pf(*this);
    }

    // The object returned by constructor can't use override operator<<.
    // So we use reference to return the reference of logger object.
    Logger& reference() {
        return *this;
    }

    virtual void outputPrefix();

    bool enableOutput() const;
    void setEnableOutput(bool enableOutput);

private:
    Severity _severity;
protected:
    std::vector<std::ostream*> _outputStreams;
private:
    std::string _sourceFile;
	std::string _sourceFileName;
    int _sourceLine;
    bool _inOutput;
    bool _enableOutput;
};

class ExpressionLogger : public Logger {
public:
    ExpressionLogger(Severity severity, std::vector<std::ostream*> outputStreams,
           const std::string& sourceFile, int sourceLine,
           bool expressionResult, const std::string& expression);
    virtual ~ExpressionLogger();

    std::string expression() const;
    void setExpression(const std::string& expression);

private:
    std::string _expression;
};

class CheckLogger : public ExpressionLogger {
public:
    CheckLogger(Severity severity, std::vector<std::ostream*> outputStreams,
           const std::string& sourceFile, int sourceLine,
           bool expressionResult, const std::string& expression);
    virtual ~CheckLogger();

    virtual void outputPrefix();

private:
    std::string _expression;
};

class LoggerStreamManager {
public:
    static LoggerStreamManager& GetInstance();

    ~LoggerStreamManager();

    std::vector<std::ostream*> getDefaultOutputStreams(Severity severity);
    void setOutputStream(std::ostream* outputStream, bool toManage = true);
    void setOutputStream(dataee::logging::Severity severity,
                         std::ostream* outputStream, bool toManage = true);
    void setOutputStream(std::vector<dataee::logging::Severity> severities,
                         std::ostream* outputStream, bool toManage = true);

private:
    LoggerStreamManager();

private:
    std::map<Severity, std::vector<std::ostream*>> _defaultOutputStreams;
    std::set<std::ostream*> _managedStreams;
};

class LogMessageVoidify {
 public:
  LogMessageVoidify() { }
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(Logger&) { }
};

}
}

const dataee::logging::Severity KLOG_DEBUG = dataee::logging::Severity::Debug;
const dataee::logging::Severity KLOG_INFO = dataee::logging::Severity::Info;
const dataee::logging::Severity KLOG_WARNING = dataee::logging::Severity::Warning;
const dataee::logging::Severity KLOG_ERROR = dataee::logging::Severity::Error;
const dataee::logging::Severity KLOG_FATAL = dataee::logging::Severity::Fatal;
