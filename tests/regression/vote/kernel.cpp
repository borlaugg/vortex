#include <vx_spawn.h>
#include "common.h"
#include <vx_intrinsics.h>

volatile int barrier_ctr;
volatile int barrier_stall;
int barrier_buffer[8];
int tls_buffer[8];
__thread int tls_var;

int __attribute__((noinline)) check_error(const int* buffer, int offset, int size) {
	int errors = 0;
	for (int i = offset; i < size; i++)	{
		int value = buffer[i];
		int ref_value = 65 + i;
		if (value == ref_value)	{
			//PRINTF("[%d] %c\n", i, value);
		} else {
			vx_printf("FAIL");
			++errors;
		}
	}
	return errors;
}

__attribute__((noinline)) void print_tls_var() {
	unsigned wid = vx_warp_id();
	tls_buffer[wid] = 65 + tls_var;
}

void tls_kernel() {
	unsigned wid = vx_warp_id();
	tls_var = wid;
	print_tls_var();
	vx_tmc(0 == wid);
}

void barrier_kernel() {
	unsigned wid = vx_warp_id();
	for (int i = 0; i <= (wid * 256); ++i) {
		++barrier_stall;
	}
	barrier_buffer[wid] = 65 + wid;
	vx_barrier(0, barrier_ctr);
	vx_tmc(0 == wid);
}


void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t count    = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_addr;
	int32_t* src1_ptr = (int32_t*)arg->src1_addr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_addr;

	uint32_t offset = blockIdx.x * count; 
	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset+i] = src0_ptr[offset+i] + src1_ptr[offset+i];
	}        
	int32_t val = 0;
    if (vx_thread_id() % 2 == 0) {
        vx_store(1,3);
		// val = 1;
    }
	else{
		vx_store(0,3);
	}
    
    // int vote_all = vx_vote_sync(0, 0, -1, val);   
	vx_vote();

	int num_warps = (vx_num_warps()< 8)?vx_num_warps():8;
	barrier_ctr = num_warps;
	barrier_stall = 0;
	vx_wspawn(num_warps, barrier_kernel);
	barrier_kernel();
	int error = check_error(barrier_buffer, 0, num_warps);

	num_warps = (vx_num_warps()< 8)?vx_num_warps():8;
	vx_wspawn(num_warps, tls_kernel);
	tls_kernel();
	check_error(tls_buffer, 0, num_warps);

}                
          
int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
      