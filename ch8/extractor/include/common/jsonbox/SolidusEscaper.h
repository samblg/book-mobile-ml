#ifndef JB_SOLIDUS_ESCAPER_H
#define JB_SOLIDUS_ESCAPER_H

#include <streambuf>

namespace jsonbox {
	class SolidusEscaper {
	public:
		SolidusEscaper();

		std::streambuf::int_type operator()(std::streambuf &destination,
		                                    std::streambuf::int_type character);
	private:
		bool afterBackSlash;
		bool inString;
	};
}

#endif
