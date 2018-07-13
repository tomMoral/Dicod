#ifndef CONVOLUTION_FFTW_H
#define CONVOLUTION_FFTW_H

#include <fftw3.h>

using namespace std;

namespace FFTW_Convolution 
{
  typedef enum
  {
    LINEAR_FULL,
    LINEAR_SAME_UNPADDED,
    LINEAR_SAME,
    LINEAR_VALID,
    CIRCULAR_SAME,
    CIRCULAR_SAME_PADDED,
    CIRCULAR_FULL_UNPADDED,
    CIRCULAR_FULL
  } Convolution_Mode;

  typedef struct Workspace
  {
    double * in_src, *out_src, *in_kernel, *out_kernel;
    int h_src, w_src, h_kernel, w_kernel;
    int w_fftw, h_fftw;
    Convolution_Mode mode;
    double * dst_fft;
    double * dst; // The array containing the result
    int h_dst, w_dst; // its size ; This is automatically set by init_workspace
    fftw_plan p_forw_src;
    fftw_plan p_forw_kernel;
    fftw_plan p_back;

  } Workspace;

  void init_workspace(Workspace & ws, Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel);
  void clear_workspace(Workspace & ws);
  // Compute the circular convolution of src and kernel modulo ws.h_fftw, ws.w_fftw
  // using the Fast Fourier Transform
  // The result is in ws.dst
  void fftw_circular_convolution(Workspace &ws, double * src, double * kernel);
  void convolve(Workspace &ws, double * src,double * kernel);


}




#endif
