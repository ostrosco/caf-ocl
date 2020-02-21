#include <clFFT.h>
#include <stdlib.h>

typedef struct cl_caf_plan_s
{
  size_t nsamps;
  float* host_buf;
  cl_mem s2_d;
  cl_mem s1_d;
  clfftPlanHandle fft_plan;
} cl_caf_plan;

cl_mem
caf_plan_get_device_buf(cl_caf_plan* caf_plan, int buf)
{
  switch (buf) {
    case 0:
      return caf_plan->s1_d;
    case 1:
      return caf_plan->s2_d;
    default:
      return NULL;
  }
}

cl_caf_plan*
create_caf_plan(cl_context ctx, cl_command_queue* queue, size_t nsamps)
{
  cl_int err;
  clfftPlanHandle fft_plan;
  clfftDim dim = CLFFT_1D;
  size_t lengths[1] = { nsamps };
  cl_caf_plan* caf_plan;

  caf_plan = (cl_caf_plan*)calloc(sizeof *caf_plan, 1);
  caf_plan->host_buf = NULL;
  caf_plan->s1_d = NULL;
  caf_plan->s2_d = NULL;

  caf_plan->s1_d = clCreateBuffer(
    ctx, CL_MEM_READ_WRITE, nsamps * 4 * sizeof(float), NULL, &err);
  caf_plan->s2_d = clCreateBuffer(
    ctx, CL_MEM_READ_WRITE, nsamps * 4 * sizeof(float), NULL, &err);
  caf_plan->host_buf = (float*)malloc(nsamps * 4 * sizeof(float));

  err = clfftCreateDefaultPlan(&fft_plan, ctx, dim, lengths);
  err = clfftSetPlanPrecision(fft_plan, CLFFT_SINGLE);
  err = clfftSetLayout(
    fft_plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
  err = clfftSetResultLocation(fft_plan, CLFFT_INPLACE);
  err = clfftBakePlan(fft_plan, 1, queue, NULL, NULL);

  caf_plan->fft_plan = fft_plan;
  caf_plan->nsamps = nsamps;

  return caf_plan;
}

void
destroy_caf_plan(cl_caf_plan* caf_plan)
{
  free(caf_plan->host_buf);
  clReleaseMemObject(caf_plan->s1_d);
  clReleaseMemObject(caf_plan->s2_d);
  free(caf_plan);
}

cl_int
execute_fft(cl_command_queue* queue,
            cl_caf_plan* caf_plan,
            int buf,
            clfftDirection direction)
{
  cl_int err;
  cl_mem buf_d;

  buf_d = caf_plan_get_device_buf(caf_plan, buf);
  if (buf_d == NULL) {
    return -1;
  }

  err = clfftEnqueueTransform(
    caf_plan->fft_plan, direction, 1, queue, 0, NULL, NULL, &buf_d, NULL, NULL);

  return err;
}

cl_int
enqueue_write_signal(cl_command_queue queue,
                     cl_caf_plan* caf_plan,
                     float* signal,
                     size_t nsamps,
                     int buf)
{
  cl_mem buf_d;
  buf_d = caf_plan_get_device_buf(caf_plan, buf);

  if (buf_d == NULL) {
    return -1;
  }

  return clEnqueueWriteBuffer(queue,
                              buf_d,
                              CL_TRUE,
                              0,
                              nsamps * 2 * sizeof(float),
                              signal,
                              0,
                              NULL,
                              NULL);
}

cl_int
enqueue_read_buffer(cl_command_queue queue, cl_caf_plan* caf_plan, int buf)
{
  cl_mem buf_d;
  buf_d = caf_plan_get_device_buf(caf_plan, buf);

  if (buf_d == NULL) {
    return -1;
  }

  return clEnqueueReadBuffer(queue,
                             buf_d,
                             CL_TRUE,
                             0,
                             caf_plan->nsamps * 4 * sizeof(float),
                             caf_plan->host_buf,
                             0,
                             NULL,
                             NULL);
}

int
main(void)
{
  cl_int err;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  float* X;
  size_t N = 16;
  cl_caf_plan* caf_plan;

  err = clGetPlatformIDs(1, &platform, NULL);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
  queue = clCreateCommandQueue(ctx, device, 0, &err);

  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  err = clfftSetup(&fftSetup);

  caf_plan = create_caf_plan(ctx, &queue, N);

  X = (float*)malloc(N * 2 * sizeof(*X));
  for (unsigned i = 0; i < N * 2; i++) {
    X[i] = (float)((i + 1) % 2);
  }

  enqueue_write_signal(queue, caf_plan, X, N, 0);
  execute_fft(&queue, caf_plan, 0, CLFFT_FORWARD);
  enqueue_read_buffer(queue, caf_plan, 0);

  clFinish(queue);
  free(X);
  destroy_caf_plan(caf_plan);
  clfftTeardown();
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
