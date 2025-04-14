#pragma once
#include <fstream>
#include <vector>

namespace IO
{

template<typename streamT, typename T>
void write_scalar_to_stream( streamT & stream, const T & scalar )
{
    stream << scalar << "\n";
}

template<typename Field_Like>
inline void write_field_to_file( const std::string & fname, const Field_Like & field )
{
    std::ofstream file( fname );
    if( file && file.is_open() )
    {
        for( const auto & scalar : field )
            write_scalar_to_stream( file, scalar );
    }
}

template<typename streamT, typename scalarT>
bool read_scalar_from_stream( streamT & stream, scalarT & scalar )
{
    const bool is_there_more_to_read = static_cast<bool>(stream >> scalar);
    return is_there_more_to_read;
}

template<typename scalarT>
std::vector<scalarT> read_field_from_file( const std::string & fname )
{
    std::ifstream infile( fname );

    scalarT temp{};
    std::vector<scalarT> result{};

    while( read_scalar_from_stream( infile, temp ) )
    {
        result.push_back( temp );
    }
    return result;
}

} // namespace IO