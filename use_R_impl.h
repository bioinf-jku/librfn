#include <R.h>

#define printf Rprintf
#define srand /* seed is done by the calling R function */
#define rand_unif unif_rand
#define rand_normal norm_rand

#include <stdexcept>
#include <sstream>

#define __num2str( x ) static_cast< std::ostringstream & >( \
   ( std::ostringstream() << std::dec << x ) ).str()

#define assert(expr) \
do \
{ \
   if(!(expr)) \
      throw std::runtime_error(std::string(#expr) + \
         std::string(" ") + std::string(__FILE__) + \
         std::string("@") + __num2str(__LINE__)); \
} \
while (0)

#define dbg_printf(...) do \
{ \
   printf(__VA_ARGS__); \
   fflush(stdout); \
} while (0)
