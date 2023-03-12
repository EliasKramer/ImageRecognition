#include "util.hpp"

std::string ms_to_string(long long ms)
{
    auto dur = std::chrono::milliseconds(ms);
    auto h = std::chrono::duration_cast<std::chrono::hours>(dur);
    auto m = std::chrono::duration_cast<std::chrono::minutes>(dur % std::chrono::hours(1));
    auto s = std::chrono::duration_cast<std::chrono::seconds>(dur % std::chrono::minutes(1));
    //auto ms_remainder = std::chrono::duration_cast<std::chrono::milliseconds>(dur % std::chrono::seconds(1));

    std::string result;

    result += std::to_string(h.count()) + "h ";
    result += std::to_string(m.count()) + "min ";
    result += std::to_string(s.count()) + "s ";
    //result += std::to_string(ms_remainder.count()) + "ms";

    return result;
}
