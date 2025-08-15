#pragma once

#include <CL/cl.h>
#include <string>
#include <vector>

class OpenCLManager {
public:
    OpenCLManager();
    ~OpenCLManager();
    
    bool initialize();
    bool createKernel(const std::string& kernelSource, const std::string& kernelName);
    bool setKernelArgs(const std::vector<void*>& args, const std::vector<size_t>& argSizes);
    bool executeKernel(size_t globalWorkSize, size_t localWorkSize);
    
    cl_mem createBuffer(size_t size, cl_mem_flags flags, void* hostPtr = nullptr);
    bool writeBuffer(cl_mem buffer, size_t size, void* data);
    bool readBuffer(cl_mem buffer, size_t size, void* data);
    void releaseBuffer(cl_mem buffer);
    
    void printDeviceInfo();
    
private:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    
    bool initialized;
    
    std::string getErrorString(cl_int error);
};