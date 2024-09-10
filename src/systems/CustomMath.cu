#include "CustomMath.cuh"

int wrap(int x, int max)
{
  int result = x % max;
  if (result < 0)
  {
    result += max;
  }
  return result;
}
